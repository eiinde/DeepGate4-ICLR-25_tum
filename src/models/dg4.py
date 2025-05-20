import torch 
import torch.nn as nn 

from .mlp import MLP
from .dg2 import DeepGate2
from .plain_tf import Plain_Transformer
from .plain_tf_linear import Sparse_Transformer
from .mlp import MLP
from .history import History
from dg_datasets.dg4_parser import AreaData
import math
from torch_scatter import scatter_add, scatter_mean, scatter_max
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.nn as gnn

from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.model_builder import GraphGymModule

#baseline method
import sys
sys.path.append('./src')
from PolarGate.model import PolarGate
import graphgps 
from DAGformer.dag_nodeformer import dag_nodeformer
import copy

_transformer_factory = {
    'baseline': None,
    'plain': Plain_Transformer,
    'sparse': Sparse_Transformer,
}

_dg2_factory = {
    'dg2': DeepGate2,
}



def build_graph(g, area_nodes, area_nodes_stats, area_faninout_cone, prob):
    area_g = AreaData()
    nodes = area_nodes[area_nodes != -1]
    pi_mask = (area_nodes_stats == 1)[:len(nodes)]
    area_g.nodes = nodes
    area_g.gate = g.gate[nodes]
    area_g.gate[pi_mask] = 0
    area_g.prob = prob[nodes]
    area_g.forward_level = g.forward_level[nodes]
    area_g.backward_level = g.backward_level[nodes]
    area_g.forward_index = torch.tensor(range(len(nodes)))
    area_g.backward_index = torch.tensor(range(len(nodes)))
    
    # Edge_index
    glo_to_area = {}
    for i, node in enumerate(nodes):
        glo_to_area[node.item()] = i
    area_edge_index = []
    for edge in g.edge_index.t():
        if edge[0].item() in glo_to_area and edge[1].item() in glo_to_area:
            area_edge_index.append([glo_to_area[edge[0].item()], glo_to_area[edge[1].item()]])
    area_edge_index = torch.tensor(area_edge_index).t()
    area_g.edge_index = area_edge_index
    
    area_g.fanin_fanout_cones = area_faninout_cone
    area_g.batch = torch.zeros(len(nodes), dtype=torch.long)
    return area_g

def merge_area_g(batch_g, g):
    no_nodes = batch_g.nodes.shape[0]
    batch_g.nodes = torch.cat([batch_g.nodes, g.nodes])
    batch_g.gate = torch.cat([batch_g.gate, g.gate])
    batch_g.prob = torch.cat([batch_g.prob, g.prob])
    batch_g.forward_level = torch.cat([batch_g.forward_level, g.forward_level])
    batch_g.backward_level = torch.cat([batch_g.backward_level, g.backward_level])
    batch_g.edge_index = torch.cat([batch_g.edge_index, g.edge_index + no_nodes], dim=1)
    batch_g.fanin_fanout_cones = torch.cat([batch_g.fanin_fanout_cones, g.fanin_fanout_cones], dim=0)
    batch_g.batch = torch.cat([batch_g.batch, torch.tensor([batch_g.batch.max() + 1] * len(g.nodes)).to(batch_g.batch.device)])
    
    batch_g.forward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    batch_g.backward_index = torch.tensor(range(len(batch_g.nodes))).to(batch_g.batch.device)
    
    return batch_g
    
def generate_orthogonal_vectors(n, dim):
    if n < dim * 8:
        # Choice 1: Generate n random orthogonal vectors in R^dim
        # Generate an initial random vector
        v0 = np.random.randn(dim)
        v0 /= np.linalg.norm(v0)
        # Generate n-1 additional vectors
        vectors = [v0]
        for i in range(n-1):
            while True:
                # Generate a random vector
                v = np.random.randn(dim)

                # Project the vector onto the subspace spanned by the previous vectors
                for j in range(i+1):
                    v -= np.dot(v, vectors[j]) * vectors[j]

                if np.linalg.norm(v) > 0:
                    # Normalize the vector
                    v /= np.linalg.norm(v)
                    break

            # Append the vector to the list
            vectors.append(v)
    else: 
        # Choice 2: Generate n random vectors:
        vectors = np.random.rand(n, dim) - 0.5
        for i in range(n):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    return vectors

def generate_hs_init(G, hs, no_dim):
    if G.batch == None:
        batch_size = 1
    else:
        batch_size = G.batch.max().item() + 1
    for batch_idx in range(batch_size):
        if G.batch == None:
            pi_mask = (G.forward_level == 0)
        else:
            pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]
        pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
        hs[pi_node] = torch.tensor(np.array(pi_vec), dtype=torch.float)
    
    return hs

class DeepGate4(nn.Module):
    def __init__(self, args, deg=None):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = args.hidden
        self.max_path_len = 257
        self.pool_depth = 2
        self.dim_feedforward = self.hidden*4
        self.tf_arch = args.tf_arch
        
        # self.tokenizer.load_pretrained(args.pretrained_model_path)

        #special token
        self.cls_token_hf = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_token_hs = nn.Parameter(torch.randn([self.hidden,]))

        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # don't learn

        self.pool_max_length = 10
        self.hf_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.hs_PositionalEmbedding = nn.Embedding(33,self.hidden)
        self.Path_Pos = nn.Embedding(self.max_path_len,self.hidden)

        # Offline Global Embedding
        self.hf_history = History(num_embeddings=10000000, embedding_dim=self.hidden)
        self.hs_history = History(num_embeddings=10000000, embedding_dim=self.hidden)

        self.hf_pooling_history = History(num_embeddings=1000000, embedding_dim=self.hidden)
        self.hs_pooling_history = History(num_embeddings=1000000, embedding_dim=self.hidden)
        self.mk = torch.zeros(10000000)
        self.hop_mk = torch.zeros(1000000)

        # Structure model
        self.sinu_pe = self.sinuous_positional_encoding(10000, self.hidden)
        self.abs_pe_embedding = nn.Linear(self.hidden, self.hidden)
        self.out_and = nn.Embedding(5000,self.hidden)
        self.out_not = nn.Embedding(5000,self.hidden)

        # Refine Transformer 
        if self.tf_arch in _transformer_factory.keys():
            # Tokenizer
            self.tokenizer = DeepGate2(dim_hidden=self.hidden)
            if args.tf_arch != 'baseline' :
                self.transformer = _transformer_factory[args.tf_arch](args, hidden=self.hidden)
        elif self.tf_arch == 'GraphGPS':
            set_cfg(cfg)
            cfg.set_new_allowed(True)
            self.args.cfg_file = './src/baseline_configs/GPS/pcqm4m-GPS.yaml'
            self.args.opts = []
            self.args.mark_done = False
            self.args.repeat = 1
            load_cfg(cfg, self.args)
            dump_cfg(cfg)
            cfg.share.dim_in = 3
            self.out_linear = nn.Linear(304,self.hidden)
            self.encoder = GraphGymModule(dim_in=cfg.share.dim_in, dim_out=self.hidden, cfg=cfg)
        elif self.tf_arch == 'Exphormer':
            set_cfg(cfg)
            cfg.set_new_allowed(True)
            self.args.cfg_file = './src/baseline_configs/Exphormer/cifar10.yaml'
            self.args.opts = []
            load_cfg(cfg, self.args)
            dump_cfg(cfg)
            cfg.share.dim_in = 3
            self.out_linear = nn.Linear(40, self.hidden)
            self.encoder = GraphGymModule(dim_in=cfg.share.dim_in, dim_out=self.hidden, cfg=cfg)
        elif self.tf_arch == 'DAGformer':
            self.encoder = dag_nodeformer(3, self.hidden, self.hidden, num_layers=4, dropout=0.1,
                        num_heads=4, use_bn=True, use_gumbel=False, use_residual=True, 
                        use_act=False, use_jk=False, rb_order=2, use_edge_loss=False)


    
        # Pooling Transformer
        pool_layer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=4, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.hop_func_tf = nn.TransformerEncoder(pool_layer, num_layers=self.pool_depth)
        self.hop_struc_tf = nn.TransformerEncoder(pool_layer, num_layers=self.pool_depth)
        
        self.init_MLP()
        self.sigmoid = nn.Sigmoid()
        
    def sinuous_positional_encoding(self, seq_len, d_model):

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))


        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe
    
    def init_MLP(self):
        self.hop_head = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=self.max_tt_len, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )         
        self.readout_prob = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_and_ratio = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_hop_level = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_level = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_num = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.connect_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_path_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_hop_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

        #Similarity
        self.gate_dis = MLP(
            dim_in=self.args.hidden*4, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_gate_ttsim = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_hop_ttsim = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_GED = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False) 

    def reset_history(self, g):
        #set to zero
        self.hf_history.reset_parameters()
        self.hs_history.reset_parameters()
        self.hf_pooling_history.reset_parameters()
        self.hs_pooling_history.reset_parameters()

        self.mk.fill_(0)
        self.hop_mk.fill_(0)

        PI_idx = g.forward_index[g.forward_level==0].cpu()

        #generate orthogonal vectors as initialized hf
        hs = generate_orthogonal_vectors(len(PI_idx), self.hidden)
        hs = torch.tensor(np.array(hs)).cpu().float()
        self.hs_history.push(hs, PI_idx)

        #init hf
        prob_mask = copy.deepcopy(g.prob)
        prob_mask = prob_mask.unsqueeze(-1)
        prob_mask = prob_mask[PI_idx]
        hf = prob_mask.repeat(1, self.hidden).clone()
        hf = hf.float()
        self.hf_history.push(hf, PI_idx)

        #init mask
        self.mk[PI_idx] = 1

    @property
    def last_shared_layer(self):
        return self.transformer.tf_layers[-1].ffn[-1]

    def forward(self, g, large_ckt=False, phase='train',avg_pooling=False):
        """
        mk: 0 denotes need to update, 1 denotes the node have been updated
        """
        device = g.gate.device
        g.nodes = g.nodes.cpu()


        if self.args.tf_arch == 'baseline' :
            avg_pooling = True

        if large_ckt==False:
            # Refine-Transformer 
            hs, hf = self.tokenizer(g, g.prob)
            if self.tf_arch != 'baseline':
                hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone(), mk=self.mk)
                #function
                hf = hf + hf_tf
                #structure
                hs = hs + hs_tf      
        else:
            hf_detach = self.hf_history.pull(g.nodes)
            hs_detach = self.hs_history.pull(g.nodes)

            if self.args.tf_arch  in _transformer_factory.keys():

                abs_pe = self.sinu_pe[g.forward_level.cpu()].to(device)
                abs_pe = self.abs_pe_embedding(abs_pe)

                #########################Local Structure########################
                #encoder the out-degree berfore structure extractor
                init_lhs = abs_pe + self.out_not(g.out_not) + self.out_and(g.out_and)
                
                ####################Global Function&Structure###################
                hs, hf = self.tokenizer(g, g.prob, hf_detach, hs_detach, mk=self.mk, lhs=init_lhs)

                if self.tf_arch!='baseline':
                    hf_tf, hs_tf = self.transformer(g, hf.clone(), hs.clone(), self.mk)
                    hf_tf = hf + hf_tf
                    hs_tf = hs + hs_tf

            elif self.args.tf_arch == 'GraphGPS':

                avg_pooling = True
                init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()
                g.x = init_emb
                g.edge_attr = torch.zeros([g.edge_index.shape[1],1]).long().to(device)
                g = self.encoder(g)
                h = self.out_linear(g.x)

                hf_tf = h.clone()
                hs_tf = h.clone()
            elif self.args.tf_arch == 'Exphormer':
                avg_pooling = True
                init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()
                g.x = init_emb
                g.edge_attr = torch.zeros([g.edge_index.shape[1],1]).to(device)
                g.num_graphs = g.batch.max()+1

                virtual_edge = g.global_virtual_edge

                virtual_edge = virtual_edge.T
                virtual_edge = virtual_edge[self.mk[g.nodes[virtual_edge[:,1].cpu()]]==0]

                g.expander_edges = virtual_edge.to(device)

                g = self.encoder(g)
                h = self.out_linear(g.x)

                hf_tf = h.clone()
                hs_tf = h.clone()

            elif self.args.tf_arch == 'DAGformer':
                avg_pooling = True

                init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()

                virtual_edge = g.global_virtual_edge

                virtual_edge = virtual_edge.T
                virtual_edge = virtual_edge[self.mk[g.nodes[virtual_edge[:,1].cpu()]]==0]
                virtual_edge = virtual_edge.T

                h = self.encoder(init_emb, [g.edge_index, virtual_edge])

                hf_tf = h.clone()
                hs_tf = h.clone()
            
            
            # each gate only update once
            update_idx = g.nodes[self.mk[g.nodes]==0].unique()

            all_update_hf = scatter(hf_tf, g.nodes.to(device), dim=0, reduce="mean")
            all_update_hs = scatter(hf_tf, g.nodes.to(device), dim=0, reduce="mean")
            
            update_hf = all_update_hf[update_idx]
            update_hs = all_update_hs[update_idx]

            hf = all_update_hf[g.nodes]
            hs = all_update_hs[g.nodes]
            

            self.hf_history.push(x = update_hf, n_id = update_idx)
            self.hs_history.push(x = update_hs, n_id = update_idx)

            self.mk[g.nodes] = 1
        
        ####################Pretrain Task########################
        if phase=='test': # to get the embedding only
            del hf,hs
            del update_hf, update_hs
            del all_update_hf, all_update_hs
            del hf_tf, hs_tf
            del hf_detach, hs_detach
            del abs_pe
            del init_lhs
            torch.cuda.empty_cache()
            return None
        
        #=========================================================
        #======================GATE-level=========================
        #=========================================================

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(update_hf)
        update_prob = F.sigmoid(prob)

        #gate-level pretrain task : predict global level
        update_level = self.readout_level(update_hs)

        #=========================================================
        #=======================pair-wise=========================
        #=========================================================
        

        tt_pair_emb = []
        tt_label = []
        dest_emb = []
        tt_pair_index = g.tt_pair_index.T[self.mk[g.tt_pair_index[0,:].cpu()]==1].T
        tt_label = g.tt_sim[self.mk[g.tt_pair_index[0,:].cpu()]==1]
        src_emb = self.hf_history.pull(tt_pair_index[0].cpu()).squeeze(-1)
        dest_emb = all_update_hf[tt_pair_index[1]]

        if dest_emb.shape[0] != 0:
            tt_pair_emb = torch.cat([src_emb, dest_emb],dim=1)
        else:
            tt_pair_emb = None
            tt_label = None

        if tt_pair_emb is None:
            pred_tt_sim = None
        elif tt_pair_emb.shape[0]==1:
            tt_pair_emb = tt_pair_emb.repeat(2,1)
            tt_label = tt_label.repeat(2)
            pred_tt_sim = self.proj_gate_ttsim(tt_pair_emb)
            pred_tt_sim = self.sigmoid(pred_tt_sim)
            
        else:
            pred_tt_sim = self.proj_gate_ttsim(tt_pair_emb)
            pred_tt_sim = self.sigmoid(pred_tt_sim)
        
            
        con_pair_emb = []
        dest_emb = []

        connect_pair_index = g.connect_pair_index.T[self.mk[g.connect_pair_index[0,:].cpu()]==1].T
        con_label = g.connect_label[self.mk[g.connect_pair_index[0,:].cpu()]==1]
        con_label = torch.where(con_label==0,0,1)
        
        src_emb = self.hs_history.pull(connect_pair_index[0,:].cpu()).squeeze(-1)
        dest_emb = all_update_hs[connect_pair_index[1]]

        if dest_emb.shape[0] != 0:
            con_pair_emb = torch.cat([src_emb,dest_emb],dim=1)
        else:
            con_pair_emb = None 
            con_label = None
        
        if con_pair_emb is None:
            pred_con = None
        elif con_pair_emb.shape[0]==1:
            con_pair_emb = con_pair_emb.repeat(2,1)
            con_label = con_label.repeat(2)
            pred_con = self.connect_head(con_pair_emb)
        else:
            pred_con = self.connect_head(con_pair_emb)
        
        #=========================================================
        #=======================HOP-level=========================
        #=========================================================


        if g.hop_idx.shape[0]==0:
            result = {
            'emb':
            {
                'hf':hf_tf,
                'hs':hs_tf,
            },
            'node':
            {
                'update_idx':update_idx,
                'prob':update_prob,
                'level':update_level,
            },
            'pair':
            {
                'pred_tt_sim':pred_tt_sim,
                'tt_label':tt_label,
                'pred_con':pred_con,
                'con_label':con_label,
            },
            'hop':None
            }
            return result
            
        ############# get pooling embedding #######################
        hop_hf = []
        hf_masks = []

        glo2loc = -1*torch.ones(g.nodes.max()+1).long()
        glo2loc[g.nodes] = g.forward_index.cpu()
        g.hop_pi = g.hop_pi.cpu()
        loc_pi = torch.where(g.hop_pi==-1,-1,glo2loc[g.hop_pi].cpu())
        pi_emb = hf[loc_pi]

        pi_emb[g.hop_pi_stats==1] = self.one_token
        pi_emb[g.hop_pi_stats==0] = self.zero_token
        pi_emb[g.hop_pi_stats==-1] = self.dc_token 
        pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
        po_emb = hf[glo2loc[g.hop_po.cpu()]]
        hop_hf = torch.cat([self.cls_token_hf.unsqueeze(0).unsqueeze(0).repeat(pi_emb.shape[0],1,1),pi_emb,po_emb], dim=1)
        if avg_pooling==True:
            pi_emb = hf[loc_pi]
            pi_emb[g.hop_pi_stats==1] = self.one_token
            pi_emb[g.hop_pi_stats==0] = self.zero_token
            pi_emb[g.hop_pi_stats==-1] = self.dc_token 
            pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
            po_emb = hf[glo2loc[g.hop_po]]
            hop_hf = torch.mean(torch.cat([pi_emb,po_emb],dim=1),dim=1)
        else:
            hf_masks = torch.where(g.hop_pi_stats==-1,1,0)
            hf_masks = torch.cat([torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device),g.hop_pi_stats,torch.zeros(g.hop_pi_stats.shape[0],1).to(hf.device)],dim=1)
            
            pos = torch.arange(hop_hf.shape[1]).unsqueeze(0).repeat(hop_hf.shape[0],1).to(hf.device)
            hop_hf = hop_hf + self.hf_PositionalEmbedding(pos)

            hf_masks = torch.where(hf_masks==1, True, False).to(hop_hf.device)

            
            hop_hf = self.hop_func_tf(hop_hf,src_key_padding_mask = hf_masks)
            hop_hf = hop_hf[:,0]

        # hop_hs = torch.stack(hop_hs) #bs seq_len hidden
        loc_nodes = torch.where(g.hop_nodes.cpu()==-1,-1,glo2loc[g.hop_nodes.cpu()])
        if avg_pooling == True :
            hop_hs = torch.mean(hs[loc_nodes],dim=1)
        else:
            hop_hs = hs[loc_nodes]
            hop_hs = torch.cat([self.cls_token_hs.reshape(1,1,-1).repeat(hop_hs.shape[0],1,1),hop_hs],dim=1)
            
            pos = torch.arange(hop_hs.shape[1]).unsqueeze(0).repeat(hop_hs.shape[0],1).to(hs.device)
            hop_hs = hop_hs + self.hs_PositionalEmbedding(pos)

            hs_masks = 1 - g.hop_nodes_stats
            hs_masks = torch.cat([torch.zeros([hs_masks.shape[0],1]),hs_masks.cpu()],dim=1)
            hs_masks = torch.where(hs_masks==1, True, False).to(hop_hs.device)

            hop_hs = self.hop_struc_tf(hop_hs,src_key_padding_mask = hs_masks)
            hop_hs = hop_hs[:,0]
        if 'name' in g.keys():
            g.hop_idx = g.hop_forward_index[:].cpu()
        self.hs_pooling_history.push(hop_hs, g.hop_idx)
        self.hf_pooling_history.push(hop_hf, g.hop_idx)
        self.hop_mk[g.hop_idx]=1
    
        # truth table prediction
        hop_tt = self.hop_head(hop_hf)
        #gate number prediction
        pred_hop_num = self.readout_num(hop_hs)
        #hop level prediction
        pred_hop_level = self.readout_hop_level(hop_hs)
        
        #=========================================================
        #================Pair-wise HOP-level======================
        #=========================================================
        #pair-wise TT sim prediction
        hop_pair = g.hop_pair_index[:,self.hop_mk[g.hop_pair_index[0].cpu()]==1].cpu()
        tt_sim_label = g.hop_tt_sim[self.hop_mk[g.hop_pair_index[0].cpu()]==1]
        hop_ged_label = g.hop_ged[self.hop_mk[g.hop_pair_index[0].cpu()]==1]
        
        if hop_pair.shape[1] == 0:
            hop_tt_sim = None
            pred_GED = None
        elif hop_pair.shape[1]==1:
            hop_pair = hop_pair.repeat(1,2)
            tt_sim_label = tt_sim_label.repeat(2)
            hop_ged_label = hop_ged_label.repeat(2)

            src_hop_hf = self.hf_pooling_history.pull(hop_pair[0].cpu())
            src_hop_hs = self.hs_pooling_history.pull(hop_pair[0].cpu())

            glo2loc = -1*torch.ones(g.hop_idx.max()+1).long()
            glo2loc[g.hop_idx] = torch.arange(g.hop_idx.shape[0])
            hop_tt_sim = self.proj_hop_ttsim(torch.cat([src_hop_hf, hop_hf[glo2loc[hop_pair[1]]]],dim=-1))
            hop_tt_sim = nn.Sigmoid()(hop_tt_sim)        
                
            #pari-wise GED prediction 
            pred_GED = self.proj_GED(torch.cat([src_hop_hs, hop_hs[glo2loc[hop_pair[1]]]],dim=-1))
            pred_GED = nn.Sigmoid()(pred_GED)
        else:
            src_hop_hf = self.hf_pooling_history.pull(hop_pair[0].cpu())
            src_hop_hs = self.hs_pooling_history.pull(hop_pair[0].cpu())

            glo2loc = -1*torch.ones(g.hop_idx.max()+1).long()
            glo2loc[g.hop_idx] = torch.arange(g.hop_idx.shape[0])
            hop_tt_sim = self.proj_hop_ttsim(torch.cat([src_hop_hf, hop_hf[glo2loc[hop_pair[1]]]],dim=-1))
            hop_tt_sim = nn.Sigmoid()(hop_tt_sim)        
                
            #pari-wise GED prediction 
            pred_GED = self.proj_GED(torch.cat([src_hop_hs, hop_hs[glo2loc[hop_pair[1]]]],dim=-1))
            pred_GED = nn.Sigmoid()(pred_GED)

        #graph-level pretrain task: on hop prediction
        glo2loc = -1*torch.ones(g.hop_idx.max()+1).long()
        glo2loc[g.hop_idx] = torch.arange(g.hop_idx.shape[0])
        ninh_mask = self.mk[g.ninh_node_index.cpu()]==1
        on_hop_emb = torch.cat([self.hs_history.pull(g.ninh_node_index[ninh_mask].cpu()),hop_hs[glo2loc[g.ninh_hop_index[ninh_mask].cpu()]]],dim=1)
        on_hop_logits = self.on_hop_head(on_hop_emb)


        result = {
            'emb':
            {
                'hf':hf_tf,
                'hs':hs_tf,
            },
            'node':
            {
                'update_idx':update_idx,
                'prob':update_prob,
                'level':update_level,
            },
            'pair':
            {
                'pred_tt_sim':pred_tt_sim,
                'tt_label':tt_label,
                'pred_con':pred_con,
                'con_label':con_label,
            },
            'hop':{
                'tt':hop_tt,
                'tt_sim':hop_tt_sim,
                'area':pred_hop_num,
                'time':pred_hop_level,
                'on_hop':on_hop_logits,
                'ninh_mask':ninh_mask,
                'GED':pred_GED,
                'tt_sim_label':tt_sim_label,
                'hop_ged_label':hop_ged_label,
            }
        }
        return result


class Baseline_Model(nn.Module):
    def __init__(self, args, deg=None):
        super().__init__()
        self.args = args
        self.max_tt_len = 64
        self.hidden = args.hidden
        self.max_path_len = 257
        self.pool_depth = 2
        self.dim_feedforward = self.hidden*4
        self.tf_arch = args.tf_arch
        self.deg = deg
        # Tokenizer
        self.encoder = self.get_encoder()

        #token for pooling
        self.cls_token_hf = nn.Parameter(torch.randn([self.hidden,]))
        self.cls_token_hs = nn.Parameter(torch.randn([self.hidden,]))

        self.dc_token = nn.Parameter(torch.randn([self.hidden,]))
        self.zero_token = nn.Parameter(torch.randn([self.hidden,]))
        self.one_token = nn.Parameter(torch.randn([self.hidden,]))
        self.pad_token = torch.zeros([self.hidden,]) # don't learn

        self.init_MLP()
        self.sigmoid = nn.Sigmoid()

    def get_encoder(self):
        encoder_type = self.args.encoder
        if encoder_type == 'PolarGate':
            return PolarGate(node_num=0, in_dim=3, out_dim=self.hidden, layer_num=9)
        elif encoder_type == 'DeepGate2':
            return DeepGate2(dim_hidden=self.hidden)
        elif encoder_type == 'GraphGPS':
            set_cfg(cfg)
            cfg.set_new_allowed(True)
            self.args.cfg_file = './src/baseline_configs/GPS/pcqm4m-GPS.yaml'
            self.args.opts = []
            self.args.mark_done = False
            self.args.repeat = 1
            load_cfg(cfg, self.args)
            dump_cfg(cfg)
            cfg.share.dim_in = 3
            self.out_linear = nn.Linear(304,self.hidden)
            return GraphGymModule(dim_in=cfg.share.dim_in, dim_out=self.hidden, cfg=cfg)
        elif encoder_type == 'Exphormer':
            set_cfg(cfg)
            cfg.set_new_allowed(True)
            self.args.cfg_file = './src/baseline_configs/Exphormer/cifar10.yaml'
            self.args.opts = []
            load_cfg(cfg, self.args)
            dump_cfg(cfg)
            cfg.share.dim_in = 3
            self.out_linear = nn.Linear(40, self.hidden)
            return GraphGymModule(dim_in=cfg.share.dim_in, dim_out=self.hidden, cfg=cfg)
        elif encoder_type == 'DAGformer':
            return dag_nodeformer(3, self.hidden, self.hidden, num_layers=4, dropout=0.1,
                        num_heads=4, use_bn=True, use_gumbel=False, use_residual=True, 
                        use_act=False, use_jk=False, rb_order=2, use_edge_loss=False)
        elif encoder_type == 'GCN':
            return gnn.GCN(in_channels=3,hidden_channels=self.hidden, num_layers=9)
        elif encoder_type == 'GraphSAGE':
            return gnn.GraphSAGE(in_channels=3,hidden_channels=self.hidden, num_layers=9)
        elif encoder_type == 'GAT':
            return gnn.GAT(in_channels=3,hidden_channels=self.hidden, num_layers=9)
        elif encoder_type == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            return gnn.PNA(in_channels=3,hidden_channels=self.hidden, num_layers=9,
                           aggregators=aggregators, scalers=scalers, deg=self.deg)
        
    
    def init_MLP(self):
        self.hop_head = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=self.max_tt_len, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )         
        self.readout_prob = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_and_ratio = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_hop_level = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_level = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_num = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.connect_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_path_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.on_hop_head = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.readout_path_len = MLP(
            dim_in=self.args.hidden, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )

        #Similarity
        self.gate_dis = MLP(
            dim_in=self.args.hidden*4, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_gate_ttsim = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_hop_ttsim = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
        self.proj_GED = MLP(
            dim_in=self.args.hidden*2, dim_hidden=self.args.mlp_hidden, dim_pred=1, 
            num_layer=self.args.mlp_layer, norm_layer=self.args.norm_layer, act_layer='relu'
        )
    
    def reset_history(self, batch):
        return

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False) 

    def forward(self, g, large_ckt=False, phase='train',avg_pooling=True):
        """
        mk: 0 denotes need to update, 1 denotes the node have been updated
        """
        device = g.gate.device

        if self.args.encoder == 'PolarGate':
            init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3)

            edge_index_s = g.edge_index
            edge_cls = g.gate[g.edge_index[1,:]].T
            edge_cls = torch.where(edge_cls==1,1,-1).to(device)
            edge_index_s = torch.cat([edge_index_s,edge_cls],dim=0)
            h = self.encoder(init_emb, edge_index_s)

            hf = h.clone()
            hs = h.clone()
        elif self.args.encoder == 'DeepGate2':
            hs, hf = self.encoder(g)
        elif self.args.encoder == 'GraphGPS':
            init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()
            g.x = init_emb
            g.edge_attr = torch.zeros([g.edge_index.shape[1],1]).long().to(device)
            g = self.encoder(g)
            h = self.out_linear(g.x)

            hf = h.clone()
            hs = h.clone()
        elif self.args.encoder == 'Exphormer':
            init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()
            g.x = init_emb
            g.edge_attr = torch.zeros([g.edge_index.shape[1],1]).to(device)
            expander_edges = []
            for areas in g.area_list[0]:
                for area in areas:
                    expander_edges.append(torch.tensor(area.nodes[area.global_virtual_edge.cpu()]))
            expander_edges = torch.cat(expander_edges,dim=1)


            g.expander_edges = g.edge_index.clone()
            g = self.encoder(g)
            h = self.out_linear(g.x)

            hf = h.clone()
            hs = h.clone()
        elif self.args.encoder == 'DAGformer':
            init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()

            expander_edges = []
            for areas in g.area_list[0]:
                for area in areas:
                    expander_edges.append(torch.tensor(area.nodes[area.global_virtual_edge.cpu()]))
            expander_edges = torch.cat(expander_edges,dim=1).to(device)

            h = self.encoder(init_emb, [g.edge_index, expander_edges])
            hf = h.clone()
            hs = h.clone()
        elif self.args.encoder in ['GCN','GraphSAGE','GAT','PNA']:
            init_emb = F.one_hot(g.gate.squeeze(-1),num_classes=3).float()
            h = self.encoder(init_emb, g.edge_index)
            hf = h.clone()
            hs = h.clone()

        if phase=='test':
            return None
        
        

        #=========================================================
        #======================GATE-level=========================
        #=========================================================

        #gate-level pretrain task : predict pari-wise TT sim
        if g.tt_pair_index.shape[1] == 0 :
            gate_tt_sim = None
        else:
            gate_tt_sim = self.proj_gate_ttsim(torch.cat([hf[g.tt_pair_index[0]],hf[g.tt_pair_index[1]]],dim=-1))
            gate_tt_sim = nn.Sigmoid()(gate_tt_sim)

        #gate-level pretrain task : predict global probability
        prob = self.readout_prob(hf)

        #gate-level pretrain task : predict global level
        pred_level = self.readout_level(hs)

        #gate-level pretrain task : predict connection
        gates = hs[g.connect_pair_index]
        gates = gates.permute(1,2,0).reshape(-1,self.hidden*2)
        pred_connect = self.connect_head(gates)


        #=========================================================
        #======================GRAPH-level========================
        #=========================================================

        pi_emb = hf[g.hop_pi]
        pi_emb[g.hop_pi_stats==1] = self.one_token
        pi_emb[g.hop_pi_stats==0] = self.zero_token
        pi_emb[g.hop_pi_stats==-1] = self.dc_token 
        pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
        po_emb = hf[g.hop_po]
        hop_hf = torch.cat([self.cls_token_hf.unsqueeze(0).unsqueeze(0).repeat(pi_emb.shape[0],1,1),pi_emb,po_emb], dim=1)
        
        pi_emb = hf[g.hop_pi]
        pi_emb[g.hop_pi_stats==1] = self.one_token
        pi_emb[g.hop_pi_stats==0] = self.zero_token
        pi_emb[g.hop_pi_stats==-1] = self.dc_token 
        pi_emb[g.hop_pi_stats==-2] = self.pad_token.to(hf.device)
        po_emb = hf[g.hop_po]
        hop_hf = torch.mean(torch.cat([pi_emb,po_emb],dim=1),dim=1)
        
        #pair-wise TT sim prediction
        hop_tt_sim = self.proj_hop_ttsim(torch.cat([hop_hf[g.hop_forward_index[g.hop_pair_index[0]]],hop_hf[g.hop_forward_index[g.hop_pair_index[1]]]],dim=-1))
        hop_tt_sim = nn.Sigmoid()(hop_tt_sim)

        # truth table prediction
        hop_tt = self.hop_head(hop_hf)

        # hop_hs = torch.stack(hop_hs) #bs seq_len hidden
        hop_hs = torch.mean(hs[g.hop_nodes],dim=1)
        
        #pari-wise GED prediction 
        pred_GED = self.proj_GED(torch.cat([hop_hs[g.hop_pair_index[0]],hop_hs[g.hop_pair_index[1]]],dim=-1))
        pred_GED = nn.Sigmoid()(pred_GED)

        #gate number prediction
        pred_hop_num = self.readout_num(hop_hs)

        #hop level prediction
        pred_hop_level = self.readout_hop_level(hop_hs)

        #graph-level pretrain task: on hop prediction
        on_hop_emb = torch.cat([hs[g.ninh_node_index],hop_hs[g.ninh_hop_index]],dim=1)
        on_hop_logits = self.on_hop_head(on_hop_emb)


        result = {
            'emb':
            {
                'hf':hf,
                'hs':hs,
            },
            'node':
            {
                'prob':prob,
                'level':pred_level,
                'connect':pred_connect,
                'tt_sim':gate_tt_sim,
            },
            'hop':{
                'tt':hop_tt,
                'tt_sim':hop_tt_sim,
                'area':pred_hop_num,
                'time':pred_hop_level,
                'on_hop':on_hop_logits,
                'GED':pred_GED,
            }
        }
        
        return result