import torch
import torch.nn.functional as F
from torch import nn

import math


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
    
class Plain_Transformer(nn.Sequential):
    def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1):
        super().__init__()
        self.args = args
        self.hidden = hidden
        self.record = {}
        self.num_head = attn_heads
        self.max_length = 511
        # self.max_length = 31
        self.dim_feedforward = 4 * self.hidden
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=self.hidden, nhead=attn_heads, dim_feedforward=self.dim_feedforward, dropout=0, batch_first=True)
        self.function_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        self.structure_transformer = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=n_layers)
        self.pos_enc = positionalencoding1d(self.hidden, self.max_length)

    def forward(self, g, hf, hs, mk=None):
        
        detach_mask = mk[g.nodes]==1
        # hf[detach_mask] = hf[detach_mask].detach()
        # hs[detach_mask] = hs[detach_mask].detach()
        device = hf.device

        self.pos_enc = self.pos_enc.to(hf.device)


        ######### use sparse attention ##################
        if 'name' in g.keys():
            g.batch = g.mini_batch[:].long()
        prefix = torch.tensor([(g.batch<b).sum() for b in g.batch.unique()])
        batch_num = prefix[:]
        bs = g.batch.max().item() + 1
        corr_m = torch.zeros(bs, self.max_length, self.max_length)
        for i in range(bs):
            if i==bs-1:
                virtual_edge = g.global_virtual_edge.T[batch_num[i]<=g.global_virtual_edge[0]].T
                virtual_edge = virtual_edge - batch_num[i]
            else:
                virtual_edge = g.global_virtual_edge.T[torch.logical_and(batch_num[i]<=g.global_virtual_edge[0],g.global_virtual_edge[0]<batch_num[i+1])].T
                virtual_edge = virtual_edge - batch_num[i]
            virtual_edge = virtual_edge.long()
            corr_m[i,virtual_edge[1],virtual_edge[0]] = 1 

        """
        False = compute attention, 
        True = mask, do not compute attention 
        """
        # for sturcture, transformer will aggregate fanin
        corr_m_hs = torch.where(corr_m == 1, False, True)
        corr_m_hf = torch.where(corr_m == 1, False, True)
    
        for i in range(bs):
            corr_m_hs[i,(g.batch==i).sum():,:] = False
            corr_m_hf[i,(g.batch==i).sum():,:] = False
            
        #multi-head attention: len, bs, emb -> len, bs*numhead, head_emb by tensor.reshape
        #corr-mask: bs,len,len
        bs,l1,l2 = corr_m_hs.shape
        corr_m_hs = corr_m_hs.unsqueeze(1).repeat(1,self.num_head,1,1).reshape(bs*self.num_head,l1,l2).to(device)
        corr_m_hf = corr_m_hf.unsqueeze(1).repeat(1,self.num_head,1,1).reshape(bs*self.num_head,l1,l2).to(device)
        
        mask_hop_states = torch.zeros([bs,self.max_length,self.hidden]).to(hf.device)
        padding_mask = torch.ones([bs,self.max_length]).to(hf.device)
        pos = torch.zeros([bs,self.max_length]).long().to(hf.device)-1
        for i in range(bs):
            mask_hop_states[i] = torch.cat([hf[g.batch==i] + hs[g.batch==i], \
                                            torch.zeros([self.max_length - hf[g.batch==i].shape[0],hf[g.batch==i].shape[1]]).to(hf.device)],dim=0)
            padding_mask[i][:hf[g.batch==i].shape[0]] = 0
            pos[i][:hf[g.batch==i].shape[0]] = g.forward_level[g.batch==i].long()

        padding_mask = torch.where(padding_mask==1, True, False)# False = compute attention, True = mask # inverse to fit nn.transformer
        
        hf_tf = self.function_transformer(mask_hop_states, src_key_padding_mask=padding_mask, mask = corr_m_hf)
        hs_tf = self.structure_transformer(mask_hop_states, src_key_padding_mask=padding_mask, mask = corr_m_hs)

        #batch to seq
        hf_tf = hf_tf.reshape([-1,self.hidden])[(padding_mask==False).reshape(-1)]
        hs_tf = hs_tf.reshape([-1,self.hidden])[(padding_mask==False).reshape(-1)]

        #only update once
        update_mask = mk[g.nodes]==0
        hf[update_mask] = hf_tf[update_mask]
        hs[update_mask] = hs_tf[update_mask]

        
        return hf, hs
