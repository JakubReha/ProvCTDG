import torch
from torch_geometric.nn import HGTConv, Linear, to_hetero, HeteroConv, TransformerConv
from typing import Callable
import math
import wandb
import time


class GraphAttentionEmbedding(torch.nn.Module):
    ''' GNN layer of the original TGN model '''

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: Callable,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = t - last_update[edge_index[0]] 
        rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))

        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class Transformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=0):
        super().__init__()
        self.conv = TransformerConv(in_channels, out_channels, heads=heads,
                                    dropout=dropout, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class HeteroGraphAttentionEmbedding(torch.nn.Module):
    ''' GNN layer of the original TGN model '''

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: Callable,
                 mean_delta_t: float = 0., std_delta_t: float = 1., metadata = None, hetero_transformer=False):
        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.metadata = metadata
        heads = 2
        self.out_dim = out_channels*heads
        if hetero_transformer:
            self.conv = HGTConv(in_channels, out_channels, metadata[1],
                           num_heads=heads, group='sum')
        else:
            self.conv = to_hetero(Transformer(in_channels, out_channels, heads=heads,
                                        dropout=0.1, edge_dim=edge_dim), metadata)
            """self.conv = [HeteroConv({
                    edge_type: TransformerConv(in_channels, out_channels, heads=heads,
                                        dropout=0.1, edge_dim=edge_dim) for edge_type in metadata[1]}, aggr='cat')]"""
            

    def forward(self, x, last_update, edge_index, t, msg):
        edge_attr = {}
        time0 = time.time()
        for edge_type in self.metadata[1]:
            rel_t = t[edge_type] - last_update[edge_type[0]][edge_index[edge_type][0]] 
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            rel_t_enc = self.time_enc(rel_t.to(x[list(x.keys())[0]].dtype))
            edge_attr[edge_type] = torch.cat([rel_t_enc, msg[edge_type]], dim=-1)
        wandb.log({"edge type for loop": time.time() - time0})
        
        return self.conv(x, edge_index, edge_attr) 
    
    

class NormalLinear(torch.nn.Linear):
    # From Jodie code
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class JodieEmbedding(torch.nn.Module):
    def __init__(self, out_channels: int,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.projector = NormalLinear(1, out_channels)

    def forward(self, x, last_update, t):
        rel_t = t - last_update
        if rel_t.shape[0] > 0:
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            return x * (1 + self.projector(rel_t.view(-1, 1).to(x.dtype))) 
    
        return x