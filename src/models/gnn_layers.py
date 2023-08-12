import torch
from torch_geometric.nn import HGTConv, Linear, to_hetero, HeteroConv, TransformerConv
from typing import Callable
import math


import os.path as osp

import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import  Linear

import math
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Parameter
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.nn import ParameterDict

from torch_geometric.typing import Adj, EdgeType, NodeType, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict


class mHGTConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        edge_feat_in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feat_in_channels = edge_feat_in_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = set([key[-1] for key in self.edge_types])

        self.edge_feat_lin = Linear(self.edge_feat_in_channels, self.out_channels//self.heads)

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels+self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.Tensor(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.Tensor(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def m_construct_bipartite_edge_index(
        self,
        edge_index_dict: Dict[EdgeType, Adj],
        src_offset_dict: Dict[EdgeType, int],
        dst_offset_dict: Dict[NodeType, int],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_feats_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Tuple[Adj, Optional[Tensor]]:
        is_sparse_tensor = False
        edge_indices: List[Tensor] = []
        edge_attrs: List[Tensor] = []
        edge_feats: List[Tensor] = []

        for edge_type, src_offset in src_offset_dict.items():
            edge_index = edge_index_dict[edge_type]
            dst_offset = dst_offset_dict[edge_type[-1]]

            # TODO Add support for SparseTensor w/o converting.
            is_sparse_tensor = isinstance(edge_index, SparseTensor)
            if is_sparse(edge_index):
                edge_index, _ = to_edge_index(edge_index)
                edge_index = edge_index.flip([0])
            else:
                edge_index = edge_index.clone()

            edge_index[0] += src_offset
            edge_index[1] += dst_offset
            edge_indices.append(edge_index)

            if edge_attr_dict is not None:
                if isinstance(edge_attr_dict, ParameterDict):
                    edge_attr = edge_attr_dict['__'.join(edge_type)]
                else:
                    edge_attr = edge_attr_dict[edge_type]
                if edge_attr.size(0) != edge_index.size(1):
                    edge_attr = edge_attr.expand(edge_index.size(1), -1)
                
                edge_attrs.append(edge_attr)
            
            if edge_feats_dict is not None:
                if isinstance(edge_feats_dict, ParameterDict):
                    edge_feat = edge_feats_dict['__'.join(edge_type)]
                else:
                    edge_feat = edge_feats_dict[edge_type]
                
                edge_feats.append(edge_feat)

        edge_index = torch.cat(edge_indices, dim=1)

        edge_attr: Optional[Tensor] = None
        if edge_attr_dict is not None:
            edge_attr = torch.cat(edge_attrs, dim=0)
        
        edge_feat: Optional[Tensor] = None
        if edge_feats_dict is not None:
            edge_feat = torch.cat(edge_feats, dim=0)

        if is_sparse_tensor:
            # TODO Add support for `SparseTensor.sparse_sizes()`.
            edge_index = SparseTensor(
                row=edge_index[1],
                col=edge_index[0],
                value=edge_attr,
            )

        return edge_index, edge_attr, edge_feat

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_features_dict: Dict[EdgeType, Tensor]  # [n_edges, edge_dim]
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr, edge_feat = self.m_construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel, edge_feats_dict=edge_features_dict)

        edge_feat = self.edge_feat_lin(edge_feat)
    
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr, edge_feat=edge_feat.repeat(self.heads, 1, 1).transpose(1, 0),
                             size=None)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k: torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor, edge_feat: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = torch.cat([v_j, edge_feat], -1) * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels+self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')


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
            self.conv = mHGTConv(in_channels, edge_dim, self.out_dim, metadata,
                           num_heads=heads, group='sum')
        else:
            self.conv = to_hetero(Transformer(in_channels, out_channels, heads=heads,
                                        dropout=0.1, edge_dim=edge_dim), metadata)
            """self.conv = [HeteroConv({
                    edge_type: TransformerConv(in_channels, out_channels, heads=heads,
                                        dropout=0.1, edge_dim=edge_dim) for edge_type in metadata[1]}, aggr='cat')]"""
            

    def forward(self, x, last_update, edge_index, t, msg):
        edge_attr = {}
        for edge_type in self.metadata[1]:
            rel_t = t[edge_type] - last_update[edge_type[0]][edge_index[edge_type][0]] 
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            rel_t_enc = self.time_enc(rel_t.to(x[list(x.keys())[0]].dtype))
            edge_attr[edge_type] = torch.cat([rel_t_enc, msg[edge_type]], dim=-1)
        
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