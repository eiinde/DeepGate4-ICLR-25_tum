
import os
import torch
from torch import nn
import time
from torch_geometric.loader import DataLoader
from deepgate.utils.logger import Logger
import torch.nn.functional as F
import sys
sys.path.append('./src')
from torch_scatter import scatter
from trainer.gradnorm import Balancer
TT_DIFF_RANGE = [0.2, 0.8]

def merge_area(batch_g):

    prefix = [0 for i in range(len(batch_g))]
    prefix = torch.zeros([len(batch_g)],dtype=torch.long)
    for i in range(prefix.shape[0]-1):
        prefix[i+1:] += batch_g[i].nodes.shape[0]

    new_batch_g = batch_g[0].clone()

    new_batch_g.nodes = torch.cat([torch.tensor(g.nodes) for g in batch_g])
    new_batch_g.po = torch.cat([g.po for g in batch_g])
    new_batch_g.gate = torch.cat([g.gate for g in batch_g])
    new_batch_g.prob = torch.cat([g.prob for g in batch_g])
    new_batch_g.forward_level = torch.cat([g.forward_level for g in batch_g])
    new_batch_g.backward_level = torch.cat([g.backward_level for g in batch_g])

    new_batch_g.edge_index = torch.cat([g.edge_index+prefix[i] for i,g in enumerate(batch_g)], dim=1)
    new_batch_g.global_virtual_edge = torch.cat([g.global_virtual_edge+prefix[i] for i,g in enumerate(batch_g)], dim=1)

    new_batch_g.batch = torch.cat([g.batch+i for i,g in enumerate(batch_g)])
    
    new_batch_g.forward_index = torch.tensor(range(len(new_batch_g.nodes)))
    new_batch_g.backward_index = torch.tensor(range(len(new_batch_g.nodes)))
    
    return new_batch_g

class Trainer():
    def __init__(self, 
                 args, 
                 model,
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 emb_dim = 128, 
                 device = 'cpu', 
                 batch_size=32, 
                 num_workers=8, 
                 distributed = False, 
                 loss = 'l2',
                 ):
        super(Trainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = emb_dim
        # self.device = device
        torch.cuda.set_device(device) #funsun: de
        self.device = torch.device('cuda') 
        self.lr = lr
        self.lr_step = -1
        self.loss_keys = [
            "l_gate_prob", "l_gate_ttsim", "l_gate_con", 
            "l_hop_tt", "l_hop_ttsim", "l_hop_GED", "l_hop_num", "l_hop_lv", "l_hop_onhop",
        ]


        # Fix loss weight
        self.loss_weight = {}
        for key in self.loss_keys:
            self.loss_weight[key] = 1.0
        
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.loss = loss
        self.hop_per_circuit = 4
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % self.args.gpus[self.local_rank]
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
            
        if self.local_rank == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.log_dir = os.path.join(save_dir, training_id)
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            # Log Path
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
            

        # Loss 
        self.consis_loss_func = nn.MSELoss().to(self.device)
        if self.loss == 'l2':
            self.loss_func = nn.MSELoss().to(self.device)
        elif self.loss == 'l1':
            self.loss_func = nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.cos_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        
        #Grad Norm
        self.loss_balancer = Balancer(self.loss_weight)

        self.model_epoch = 0
        
        # Temp Data 
        self.stru_sim_tmp = {}
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
            
        # Resume 
        if self.args.resume:
            stats = self.resume()
            assert stats
    
    def set_training_args(self, loss_weight={}, lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) > 0:
            for key in loss_weight:
                self.loss_weight[key] = loss_weight[key]
                print('[INFO] Update {} weight from {}'.format(key, loss_weight[key]))
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            self.optimizer = self.optimizer


    def save(self, filename):
        path = os.path.join(self.log_dir, filename)
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if self.local_rank == 0:
            print('[INFO] Load checkpoint from: ', model_path)
        if os.path.exists(model_path):
            self.load(model_path)
            return True
        else:
            return False

    def run_batch(self, batch, phase='train'):
        
        t0 = time.time()

        result_dict = self.model(batch, large_ckt=self.args.enable_cut, phase=phase)
        
        runtime = time.time() - t0
        #=========================================================
        #======================GATE-level=========================
        #=========================================================  
        update_idx = result_dict['node']['update_idx']

        device = batch.prob.device
        label_idx = scatter(batch.forward_index, batch.nodes.to(device), dim=0, reduce="min")[update_idx]

        # logic probility predction(gate-level)
        l_gate_prob = self.l1_loss(result_dict['node']['prob'], batch.prob[label_idx].unsqueeze(1))
        
        #pair-wise
        if result_dict['pair']['pred_tt_sim'] is not None:
            l_gate_sim = self.l1_loss(result_dict['pair']['pred_tt_sim'].squeeze(-1), result_dict['pair']['tt_label'].to(device))
        else:
            l_gate_sim = None

        #connect classification
        if result_dict['pair']['pred_con'] is not None:
            prob = F.sigmoid(result_dict['pair']['pred_con']).squeeze(-1)
            l_gate_con = self.bce(prob,result_dict['pair']['con_label'].float().to(device))
            pred_cls = torch.where(prob > 0.5, 1, 0).cpu()
            con_acc = torch.sum(pred_cls==result_dict['pair']['con_label'].cpu()) * 1.0 / prob.shape[0]
        else:
            l_gate_con = None
            con_acc = None

        if result_dict['hop'] is None:
            l_hop_tt = None
            l_hop_ttsim = None
            l_hop_GED = None
            l_hop_num = None
            l_hop_lv = None
            l_hop_onhop = None
            on_hop_acc = None
            hamming_dist = None
        else:
            try:
                pred_hop_tt_prob = nn.Sigmoid()(result_dict['hop']['tt']).to(self.device)
            except TypeError:
                print(1)
            pred_tt = torch.where(pred_hop_tt_prob > 0.5, 1, 0)
            pred_hop_tt_prob = torch.clamp(pred_hop_tt_prob, 1e-6, 1-1e-6)
            l_hop_tt = self.bce(pred_hop_tt_prob, batch.hop_tt.float())
            hamming_dist = torch.mean(torch.abs(pred_tt.float()-batch.hop_tt.float())).cpu()


            
            if result_dict['hop']['tt_sim'] is not None:
                # pair-wise tt sim
                pred_hop_ttsim = result_dict['hop']['tt_sim'].squeeze(-1)
                l_hop_ttsim = self.l1_loss(pred_hop_ttsim, result_dict['hop']['tt_sim_label'].to(self.device))

                #pair wise GED
                pred_hop_GED = result_dict['hop']['GED'].squeeze(-1)
                l_hop_GED = self.l1_loss(pred_hop_GED, result_dict['hop']['hop_ged_label'].to(self.device))
            else:
                l_hop_ttsim = None
                l_hop_GED = None
                hamming_dist = None

            #hop num prediction
            l_hop_num = self.l1_loss(result_dict['hop']['area'].squeeze(-1), torch.sum(batch.hop_nodes_stats,dim=1).to(self.device))

            #hop level prediction
            l_hop_lv = self.l1_loss(result_dict['hop']['time'].squeeze(-1), batch.hop_levs.to(self.device))

            #hop on-hop prediction
            pred_on_hop_prob = nn.Sigmoid()(result_dict['hop']['on_hop']).squeeze(-1).to(self.device)
            l_hop_onhop = self.bce(pred_on_hop_prob,batch.ninh_labels[result_dict['hop']['ninh_mask']].float())
            pred_on_hop_label = torch.where(pred_on_hop_prob>0.5,1,0)
            on_hop_acc = (pred_on_hop_label==batch.ninh_labels[result_dict['hop']['ninh_mask']]).sum()*1.0/pred_on_hop_label.shape[0]


        if self.args.skip_hop:
            l_hop_tt = None
            l_hop_ttsim = None
            l_hop_GED = None
            l_hop_num = None
            l_hop_lv = None
            l_hop_onhop = None
            on_hop_acc = None
            hamming_dist = None
            # # Loss 
            func_loss = 0
            for l in [l_gate_prob,l_gate_sim]:
                if l is not None:
                    func_loss+=l
            if l_gate_con is not None:
                stru_loss =  l_gate_con
            else:
                stru_loss = 0
            loss_status = {
            #gate-level
            'l_gate_prob': l_gate_prob,
            'l_gate_ttsim': l_gate_sim,
            'l_gate_con': l_gate_con,
            "con_acc": con_acc,

            #hop-level
            'l_hop_tt':l_hop_tt,
            'l_hop_ttsim':l_hop_ttsim,
            'l_hop_GED':l_hop_GED,
            'l_hop_num':l_hop_num,
            'l_hop_lv':l_hop_lv,
            'l_hop_onhop':l_hop_onhop,
            'on_hop_acc':on_hop_acc,
            'hamming_dist':hamming_dist,

            #overall
            'loss' : func_loss+stru_loss,
            'func_loss':func_loss,
            'stru_loss':stru_loss,
        }
        else:

            loss_status = {
            #gate-level
            'l_gate_prob': l_gate_prob,
            'l_gate_ttsim': l_gate_sim,
            'l_gate_con': l_gate_con,
            "con_acc": con_acc,

            #hop-level
            'l_hop_tt':l_hop_tt,
            'l_hop_ttsim':l_hop_ttsim,
            'l_hop_GED':l_hop_GED,
            'l_hop_num':l_hop_num,
            'l_hop_lv':l_hop_lv,
            'l_hop_onhop':l_hop_onhop,
            'on_hop_acc':on_hop_acc,
            'hamming_dist':hamming_dist,

            #overall
            'loss' : 0,
            'func_loss':0,
            'stru_loss':0,

            #time
            'runtime':runtime
        }
            # Loss 
            for k in ['l_gate_prob','l_gate_ttsim','l_hop_tt','l_hop_ttsim']:
                if loss_status[k] is not None:
                    loss_status['func_loss']+=loss_status[k] * self.loss_weight[k]
            for k in ['l_gate_con','l_hop_GED','l_hop_num','l_hop_lv','l_hop_onhop']:
                if loss_status[k] is not None:
                    loss_status['stru_loss']+=loss_status[k]* self.loss_weight[k]
            loss_status['loss'] = loss_status['func_loss'] + loss_status['stru_loss']

        # h = torch.cat([result_dict['emb']['hf'],result_dict['emb']['hs']],dim=-1)
        return loss_status, result_dict['emb']['hf'], result_dict['emb']['hs']
    
    def run_batch_baseline(self, batch, phase='train'):
        
        t0 = time.time()
        result_dict = self.model(batch, large_ckt=self.args.enable_cut, phase=phase)
        runtime = time.time()-t0
        #=========================================================
        #======================GATE-level=========================
        #=========================================================  
             
        # logic probility predction(gate-level)
        l_gate_prob = self.l1_loss(result_dict['node']['prob'], batch.prob.unsqueeze(1).to(self.device))
        # gate level prediction
        #connect classification
        prob = F.sigmoid(result_dict['node']['connect']).squeeze(-1)
        con_label = torch.where(batch.connect_label==0,0,1).float()
        l_gate_con = self.bce(prob,con_label)

        pred_cls = torch.where(prob > 0.5, 1, 0)
        con_acc = torch.sum(pred_cls==con_label) * 1.0 / prob.shape[0]

        # pred_tt_sim = (result_dict['node']['tt_sim']+1)/2
        if result_dict['node']['tt_sim']is not None:
            l_gate_ttsim = self.l1_loss(result_dict['node']['tt_sim'].squeeze(-1), batch.tt_sim.to(self.device))
        else:
            l_gate_ttsim = None


        #=========================================================
        #======================GRAPH-level========================
        #=========================================================

        # Truth table predction(graph-level)
        pred_hop_tt_prob = nn.Sigmoid()(result_dict['hop']['tt']).to(self.device)
        pred_tt = torch.where(pred_hop_tt_prob > 0.5, 1, 0)
        pred_hop_tt_prob = torch.clamp(pred_hop_tt_prob, 1e-6, 1-1e-6)
        l_hop_tt = self.bce(pred_hop_tt_prob, batch.hop_tt.float())
        hamming_dist = torch.mean(torch.abs(pred_tt.float()-batch.hop_tt.float())).cpu()

        # pred_hop_ttsim = (result_dict['hop']['tt_sim'].squeeze(-1)+1)/2
        pred_hop_ttsim = result_dict['hop']['tt_sim'].squeeze(-1)
        l_hop_ttsim = self.l1_loss(pred_hop_ttsim, batch.hop_tt_sim.to(self.device))


        # pred_hop_GED = (result_dict['hop']['GED'].squeeze(-1)+1)/2
        pred_hop_GED = result_dict['hop']['GED'].squeeze(-1)
        l_hop_GED = self.l1_loss(pred_hop_GED, batch.hop_ged.to(self.device))

        #hop num prediction
        l_hop_num = self.l1_loss(result_dict['hop']['area'].squeeze(-1), torch.sum(batch.hop_nodes_stats,dim=1).to(self.device))

        #hop level prediction
        l_hop_lv = self.l1_loss(result_dict['hop']['time'].squeeze(-1), batch.hop_levs.to(self.device))

        #hop on-hop prediction
        pred_on_hop_prob = nn.Sigmoid()(result_dict['hop']['on_hop']).squeeze(-1).to(self.device)
        l_hop_onhop = self.bce(pred_on_hop_prob,batch.ninh_labels.float())
        pred_on_hop_label = torch.where(pred_on_hop_prob>0.5,1,0)
        on_hop_acc = (pred_on_hop_label==batch.ninh_labels).sum()*1.0/pred_on_hop_label.shape[0]


        loss_status = {
            #gate-level
            'l_gate_prob': l_gate_prob,
            'l_gate_ttsim': l_gate_ttsim,
            'l_gate_con': l_gate_con,
            "con_acc": con_acc,

            #hop-level
            'l_hop_tt':l_hop_tt,
            'l_hop_ttsim':l_hop_ttsim,
            'l_hop_GED':l_hop_GED,
            'l_hop_num':l_hop_num,
            'l_hop_lv':l_hop_lv,
            'l_hop_onhop':l_hop_onhop,
            'on_hop_acc':on_hop_acc,
            'hamming_dist':hamming_dist,

            #overall
            'loss' : 0,
            'func_loss':0,
            'stru_loss':0,

            #time
            'runtime':runtime,
        }
            # Loss 
        for k in ['l_gate_prob','l_gate_ttsim','l_hop_tt','l_hop_ttsim']:
            if loss_status[k] is not None:
                loss_status['func_loss']+=loss_status[k] * self.loss_weight[k]
        for k in ['l_gate_con','l_hop_GED','l_hop_num','l_hop_lv','l_hop_onhop']:
            if loss_status[k] is not None:
                loss_status['stru_loss']+=loss_status[k]* self.loss_weight[k]
        loss_status['loss'] = loss_status['func_loss'] + loss_status['stru_loss']

        return loss_status
        
    
    def run_dataset(self, epoch, dataset, phase='train'):
        overall_dict = {
            #gate-level
            'l_gate_prob': [],
            'l_gate_ttsim':[],
            'l_gate_con':[],
            'con_acc':[],

            #hop-level
            "l_hop_tt":[],
            "l_hop_ttsim":[],
            "l_hop_GED":[],
            "l_hop_num":[],
            "l_hop_lv":[],
            "l_hop_onhop":[],
            "hamming_dist":[],
            "on_hop_acc":[],

            #overall
            'loss':[],
            'func_loss':[],
            'stru_loss':[],

            #time
            'runtime':[],
        }

        mini_batch_size = self.args.mini_batch_size

        for iter_id, batch in enumerate(dataset):

            if self.local_rank == 0:
                time_stamp = time.time()
            self.model.reset_history(batch)

            # without parition, for baseline method
            if self.args.enable_cut==False:
                area_lv_i = 0
                batch = batch.to(self.device) 
                if phase=='test':
                    _ = self.model(batch, large_ckt=self.args.enable_cut,phase=phase)
                else:
                    loss_dict = self.run_batch_baseline(batch,phase=phase)

                    for loss_key in loss_dict:
                        if loss_key=='con_acc' or loss_key=='hamming_dist' or loss_key=='on_hop_acc':
                            if loss_dict[loss_key] is not None:
                                overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu())
                        if loss_key=='runtime':
                            overall_dict[loss_key].append(loss_dict[loss_key])
                        else:
                            if loss_dict[loss_key] is not None:
                                overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu().item())

                    loss = loss_dict['loss']

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            # with parition, for DeepGate4
            else:
                if 'merge_small' in batch.name[0]:
                    #run batch
                    area_lv_i = 0
                    mem_list = []
                    batch = batch.to(self.device)
                    batch.global_virtual_edge = batch.global_virtual_edge.long()
                    batch.hop_idx = torch.arange(batch.hop_nodes.shape[0])

                    loss_dict, hf, hs = self.run_batch(batch,phase=phase)

                    assert torch.isnan(loss_dict['loss'])==False, f"{batch.name, area_lv_i}: loss is nan"

                    for loss_key in loss_dict:
                        if loss_key=='con_acc' or loss_key=='hamming_dist' or loss_key=='on_hop_acc':
                            if loss_dict[loss_key] is not None:
                                overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu())
                        if loss_key=='runtime':
                            overall_dict[loss_key].append(loss_dict[loss_key])
                        else:
                            if loss_dict[loss_key] is not None:
                                overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu().item())

                    losses = {k:loss_dict[k] for k in self.loss_keys}

                    if self.args.tf_arch == 'sparse':
                        if phase == 'train':
                            loss = self.loss_balancer.backward(losses,self.model.last_shared_layer)
                        else:
                            loss = loss_dict['loss']
                    else:
                        loss = loss_dict['loss']

                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        self.optimizer.step()
                        
                else:
                    max_area_lv = len(batch.area_list[0])
                    if phase!='test':
                        batch.tt_pair_index = batch.tt_pair_index.to(self.device)
                        batch.tt_sim = batch.tt_sim.to(self.device)
                        batch.connect_pair_index = batch.connect_pair_index.to(self.device)
                        batch.connect_label = batch.connect_label.to(self.device)

                    for area_lv_i in range(max_area_lv):

                        area_num = len(batch.area_list[0][area_lv_i])
                        mini_batch_num = (area_num - 1) // mini_batch_size + 1
                        
                        for mini_batch_id in range(mini_batch_num):
                            if mini_batch_id != mini_batch_num - 1 :
                                mini_batch = batch.area_list[0][area_lv_i][mini_batch_id*mini_batch_size : (mini_batch_id+1)*mini_batch_size] 
                            else:
                                mini_batch = batch.area_list[0][area_lv_i][mini_batch_id*mini_batch_size : ]
                            mini_batch = merge_area(mini_batch)


                            mini_batch.gate = batch.gate[mini_batch.nodes.cpu()]
                            
                            mini_batch = mini_batch.to(self.device) 
                            mini_batch.out_and = batch.out_and[mini_batch.nodes.cpu()].to(self.device)
                            mini_batch.out_not = batch.out_not[mini_batch.nodes.cpu()].to(self.device)
                            if phase!='test':
                                
                                #add pair wise label
                                mini_batch.tt_pair_index = batch.tt_pair_index[:,torch.isin(batch.tt_pair_index[1].to(self.device),mini_batch.nodes)]
                                mini_batch.tt_sim = batch.tt_sim[torch.isin(batch.tt_pair_index[1].to(self.device),mini_batch.nodes)]

                                mini_batch.connect_pair_index = batch.connect_pair_index[:,torch.isin(batch.connect_pair_index[1].to(self.device),mini_batch.nodes)]
                                mini_batch.connect_label = batch.connect_label[torch.isin(batch.connect_pair_index[1].to(self.device),mini_batch.nodes)]

                                #add hop level label
                                hop_idx = torch.isin(batch.hop_po.to(self.device)[:,0],mini_batch.po).cpu()
                                mini_batch.hop_idx = torch.argwhere(hop_idx==True).squeeze(-1)
                                mini_batch.hop_nodes = batch.hop_nodes[hop_idx]
                                mini_batch.hop_pi = batch.hop_pi[hop_idx]
                                mini_batch.hop_po = batch.hop_po[hop_idx]
                                mini_batch.hop_pi_stats = batch.hop_pi_stats[hop_idx].to(self.device)
                                mini_batch.hop_forward_index = batch.hop_forward_index[hop_idx]  
                                mini_batch.hop_nodes_stats = batch.hop_nodes_stats[hop_idx]
                                mini_batch.hop_forward_index = batch.hop_forward_index[hop_idx]
                                mini_batch.hop_tt = batch.hop_tt[hop_idx].to(self.device)
                                mini_batch.hop_levs = batch.hop_levs[hop_idx].to(self.device)
                                mini_batch.hop_nds = batch.hop_nds[hop_idx].to(self.device)
                                

                                # add pair-wise hop level label
                                pair_idx = torch.isin(batch.hop_po[batch.hop_pair_index[1]].squeeze(-1).to(self.device),mini_batch.po).cpu()
                                mini_batch.hop_pair_index = batch.hop_pair_index[:,pair_idx].to(self.device)
                                mini_batch.hop_tt_sim = batch.hop_tt_sim[pair_idx].to(self.device)
                                mini_batch.hop_ged = batch.hop_ged[pair_idx].to(self.device)

                                ninh_pair_idx = (hop_idx[batch.ninh_hop_index]==True).cpu()
                                mini_batch.ninh_node_index = batch.ninh_node_index[ninh_pair_idx]
                                mini_batch.ninh_hop_index = batch.ninh_hop_index[ninh_pair_idx]
                                mini_batch.ninh_labels = batch.ninh_labels[ninh_pair_idx].to(self.device)
                                
                                #run batch
                                loss_dict, hf, hs = self.run_batch(mini_batch,phase=phase)


                                assert torch.isnan(loss_dict['loss'])==False, f"{batch.name, area_lv_i}: loss is nan"

                                for loss_key in loss_dict:
                                    if loss_key=='con_acc' or loss_key=='hamming_dist' or loss_key=='on_hop_acc':
                                        if loss_dict[loss_key] is not None:
                                            overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu())
                                    elif loss_key=='runtime':
                                        overall_dict[loss_key].append(loss_dict[loss_key])
                                    else:
                                        if loss_dict[loss_key] is not None:
                                            overall_dict[loss_key].append(loss_dict[loss_key].detach().cpu().item())

                                losses = {k:loss_dict[k] for k in self.loss_keys}

                                if self.args.tf_arch == 'sparse':
                                    loss = self.loss_balancer.backward(losses,self.model.last_shared_layer)
                                else:
                                    loss = loss_dict['loss']

                            else:
                                _ = self.model(mini_batch,  large_ckt=self.args.enable_cut,phase=phase)
                                del mini_batch
                                torch.cuda.empty_cache()

                            if phase == 'train':
                                self.optimizer.zero_grad()
                                loss.backward(retain_graph=True)
                                self.optimizer.step()

            if phase=='train':
                if self.local_rank == 0:
                    output_log = '({phase}) Epoch: {epoch} | Iter: {iter} | Circuit:{name} | Area_lv:{area_lv_i} | Time: {time:.4f} '.format(
                        phase=phase, epoch=epoch, iter=iter_id,name=batch.name[0],area_lv_i=area_lv_i, time=time.time()-time_stamp
                    )
                    output_log += '\n======================GATE-level======================== \n'
                    gate_loss = 0
                    for loss_key in loss_dict:
                        if 'l_gate' in loss_key:
                            if loss_dict[loss_key] is None:
                                l = 0.
                            else:
                                l = loss_dict[loss_key]
                            output_log += ' | {}: {:.4f}'.format(loss_key, l)
                            gate_loss += l
                    output_log += ' | {}: {:.4f}'.format('overall loss', gate_loss)

                    output_log += '\n======================HOP-level======================== \n'
                    hop_loss = 0
                    for loss_key in loss_dict:
                        if 'l_hop' in loss_key:                                                                
                            if loss_dict[loss_key] is None:
                                l = 0.
                            else:
                                l = loss_dict[loss_key]
                            output_log += ' | {}: {:.4f}'.format(loss_key, l)
                            hop_loss += l
                    output_log += ' | {}: {:.4f}'.format('overall loss', gate_loss)

                    output_log += '\n======================All-level========================= \n'
                    overall_keys = ['loss','func_loss','stru_loss','con_acc','hamming_dist','on_hop_acc']
                    for k in overall_keys:
                        if loss_dict[k] is not None:
                            l = loss_dict[k]
                        else:
                            l=0
                        output_log += ' | {}: {:.4f}'.format(k, l)


                    print(output_log)
                    print('\n')           

        if self.local_rank == 0:
            for k in overall_dict:
                print('overall {}:{:.4f}'.format(k,torch.mean(torch.tensor(overall_dict[k]).float())))
            print('\n')

        return 0, 0, torch.mean(torch.tensor(overall_dict['func_loss']).float()), \
        torch.mean(torch.tensor(overall_dict['stru_loss']).float()), torch.mean(torch.tensor(overall_dict['loss']).float()),
    
    def train(self, num_epoch, train_dataset, val_dataset):
        
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=self.num_workers)
            
        if self.distributed: 
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=val_sampler)
        else:
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        
        for epoch in range(num_epoch):
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    self.model.to(self.device)
                    
                    self.run_dataset(epoch, train_dataset, phase)
                else:
                    self.model.eval()
                    self.model.to(self.device)
                    self.run_dataset(epoch, val_dataset, phase)

            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
            # Save model 
            if self.local_rank == 0:
                self.save('model_last.pth')
                if epoch % 10 == 0:
                    self.save('model_{:}.pth'.format(epoch))
                    print('[INFO] Save model to: ', os.path.join(self.log_dir, 'model_{:}.pth'.format(epoch)))
                    
    def inference(self, val_dataset):

        val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        self.model.eval()
        self.model.to(self.device)

        _, _, func, stru, overall = self.run_dataset(0, val_dataset, phase='test')


        return 0, 0, func, stru, overall