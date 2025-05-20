import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, LayerNorm

class GATTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, concat=True, dropout=0.1, ff_hidden_dim=128):
        super(GATTransformerEncoderLayer, self).__init__()
        
        # GAT multi-head attention
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat)
        
        # Feed-forward network (FFN)
        self.ffn = torch.nn.Sequential(
            Linear(out_channels*heads if concat else out_channels, ff_hidden_dim),
            torch.nn.ReLU(),
            Linear(ff_hidden_dim, out_channels*heads if concat else out_channels)
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(out_channels*heads if concat else out_channels)
        self.norm2 = LayerNorm(out_channels*heads if concat else out_channels)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # GAT layer with residual connection
        x_residual = x.clone()
        x = self.gat(x, edge_index)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm1(x)   # Layer normalization
        
        # Feed-forward network with residual connection
        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + x_residual  # Residual connection
        x = self.norm2(x)   # Layer normalization
        
        return x



class Sparse_Transformer(torch.nn.Module):
    def __init__(self, args, hidden, num_layers=12, heads=4, concat=True, dropout=0.1):
        super(Sparse_Transformer, self).__init__()
 
        in_channels = hidden * 2
        out_channels = in_channels // heads

        ff_hidden_dim = 4 * hidden

        self.num_layers = num_layers

        self.tf_layers = torch.nn.ModuleList([
            GATTransformerEncoderLayer(in_channels if i == 0 else out_channels*heads if concat else out_channels,
                                       out_channels, heads=heads, concat=concat, dropout=dropout, ff_hidden_dim=ff_hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, g, hf, hs, mk):
    
        virtual_edge = g.global_virtual_edge

        virtual_edge = virtual_edge.T
        virtual_edge = virtual_edge[mk[g.nodes[virtual_edge[:,1].cpu()]]==0]
        virtual_edge = virtual_edge.T

        if virtual_edge.shape[1] == 0:
            return hf, hs
        
        h = torch.cat([hf,hs],dim=-1)
        for i in range(self.num_layers):
            h = self.tf_layers[i](h,virtual_edge)
    
        hf, hs = torch.chunk(h,2,dim=-1)

        return hf, hs
