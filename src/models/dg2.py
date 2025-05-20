import deepgate as dg 
import torch
import copy
import numpy as np 
from .mlp import MLP
import torch.nn.init as init
import torch.nn as nn
from torch.nn import GRU
import os
from .tfmlp import TFMlpAggr

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
    
    return hs, -1

def get_slices(G, mk = None):
    device = G.gate.device
    edge_index = G.edge_index
    
    
    # # Edge slices 
    # edge_level = torch.index_select(G.forward_level, dim=0, index=edge_index[1])
    # # sort edge according to level
    # edge_indices = torch.argsort(edge_level)
    # edge_index = edge_index[:, edge_indices]
    # edge_level_cnt = torch.bincount(edge_level).tolist()
    # edge_index_slices = torch.split(edge_index, list(edge_level_cnt), dim=1)
    
    # Index slices
    and_index_slices = []
    not_index_slices = []
    edge_index_slices = []
    and_mask = (G.gate == 1).squeeze(1)
    not_mask = (G.gate == 2).squeeze(1)

    # one gate will only be updated once
    if mk is not None:
        mk = mk.to(device)
        and_mask = torch.logical_and(and_mask,mk[G.nodes]==0)
        not_mask = torch.logical_and(not_mask,mk[G.nodes]==0)
        
    for level in range(0, torch.max(G.forward_level).item() + 1):
        and_level_nodes = torch.nonzero((G.forward_level == level) & and_mask).squeeze(1)
        not_level_nodes = torch.nonzero((G.forward_level == level) & not_mask).squeeze(1)
        and_index_slices.append(and_level_nodes)
        not_index_slices.append(not_level_nodes)
        edge_index_slices.append(edge_index[:,G.forward_level[edge_index[1]]==level])
    
    return and_index_slices, not_index_slices, edge_index_slices

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

class DeepGate2(nn.Module):
    def __init__(self, num_rounds=1, dim_hidden=128, enable_encode=True, enable_reverse=False):
        super(DeepGate2, self).__init__()

        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # Network 
        self.aggr_and_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_not_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_and_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
            
        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)

        # Readout 
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')

        # self.encoder = MLP(
        #     dim_in=dim_hidden, dim_hidden=dim_hidden*4, dim_pred=dim_hidden, 
        #     num_layer=3, norm_layer='batchnorm', act_layer='relu'
        # )
        # torch.nn.init.xavier_uniform_(self.aggr_and_strc.weight)
        # torch.nn.init.xavier_uniform_(self.aggr_not_strc.weight)
        # torch.nn.init.xavier_uniform_(self.aggr_and_func.weight)
        # torch.nn.init.xavier_uniform_(self.aggr_not_func.weight)
        # initialize_weights(self.aggr_and_strc)
        # initialize_weights(self.aggr_not_strc)
        # initialize_weights(self.aggr_and_func)
        # initialize_weights(self.aggr_not_func)
        # self.hs0 = nn.Embedding(1,dim_hidden)
    
    
    def forward(self, G, PI_prob=None,hf_glo=None,hs_glo=None,lhs=None,mk=None):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        max_num_layers = torch.max(G.forward_level).item() + 1
        min_num_layers = G.forward_level.min()
        
        hs = torch.zeros(num_nodes, self.dim_hidden)
        hf = torch.zeros(num_nodes, self.dim_hidden)

        hs = hs.to(device)
        hf = hf.to(device)

        #substitute emb with global emb
        #modifed by 
        if hf_glo is not None:
            update_mask = mk[G.nodes.cpu()]==1

            hs[update_mask] = hs_glo[update_mask]
            hf[update_mask] = hf_glo[update_mask]
        
        if lhs is not None:
            unupdate_mask = mk[G.nodes.cpu()]==0
            hs[unupdate_mask] = hs[unupdate_mask] + lhs[unupdate_mask]
        
        node_state = torch.cat([hs, hf], dim=-1)
            
        
        and_slices, not_slices, edge_slices = get_slices(G, mk)

        for _ in range(self.num_rounds):

            # Add virtual AND gate to get better initial embeding
                    # Add 2 not gate for PI to get better initial embeding 
            PIs = G.forward_index[G.forward_level==0]

            #=========================
            if PIs.shape!=0:

                hf_new = torch.cat([hf,torch.zeros(PIs.shape[0]*2,hf.shape[1]).to(device)])
                # hs_new = torch.cat([hs,torch.zeros(PIs.shape[0]*2,hf.shape[1]).to(device)])

                # vitual not gate1
                virtue_not_index1 = torch.arange(hf.shape[0],hf.shape[0]+PIs.shape[0]).to(device)
                not_edge_index1 = torch.stack([PIs,virtue_not_index1])
                msg = self.aggr_not_func(hf_new, not_edge_index1)
                not_msg = torch.index_select(msg, dim=0, index=virtue_not_index1)
                hf_not = torch.index_select(hf_new, dim=0, index=virtue_not_index1)
                _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                hf_new[virtue_not_index1, :] = hf_not.squeeze(0)

                #vitual not gate2
                virtue_not_index2 = torch.arange(hf.shape[0]+PIs.shape[0],hf.shape[0]+PIs.shape[0]*2).to(device)
                not_edge_index2 = torch.stack([virtue_not_index1, virtue_not_index2])
                msg = self.aggr_not_func(hf_new, not_edge_index2)
                not_msg = torch.index_select(msg, dim=0, index=virtue_not_index2)
                hf_not = torch.index_select(hf_new, dim=0, index=virtue_not_index2)
                _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                hf_new[virtue_not_index2, :] = hf_not.squeeze(0)

                hf[PIs] = hf_new[virtue_not_index2]

                node_state = torch.cat([hs, hf], dim=-1)
                # node_state = torch.cat([hs+lhs, hf], dim=-1)

            for level in range(min_num_layers, max_num_layers):
                l_and_node = and_slices[level]

                if l_and_node.size(0) > 0:
                    and_edge_index = edge_slices[level]
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate
                l_not_node = not_slices[level]
                if l_not_node.size(0) > 0:
                    not_edge_index = edge_slices[level]
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_not_func(hf, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)
                
                # Update node state

                node_state = torch.cat([hs, hf], dim=-1)

        hs = node_state[:, :self.dim_hidden]
        hf = node_state[:, self.dim_hidden:]

        return hs, hf

    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
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
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)