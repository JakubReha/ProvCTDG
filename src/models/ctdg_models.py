import torch
from torch_geometric.nn.models.tgn import TimeEncoder, LastAggregator
from torch_geometric.nn import to_hetero
from torch_geometric.nn.resolver import activation_resolver
from typing import Callable, Optional, Any, Dict, Union, List, Tuple, Set
from .predictors import *
from .memory_layers import *
from .gnn_layers import *
from .message_aggregators import AGGREGATOR_CONFS, IdentityMessage
import time
import wandb
import numpy as np



def get_indices(ntypes: Tensor, n_ids: Tensor, available_ntypes=(0, 1, 2)):
    type_to_id = dict()
    old_to_new = dict()
    for _type in available_ntypes:
        # integer indexes of nodes that have type `_type`
        _mask: Tensor[int] = torch.nonzero(ntypes==_type)
        type_to_id[_type] = _mask.view(-1).tolist()
        old_to_new.update({n_id_old.item(): n_id_new for n_id_new, n_id_old in enumerate(n_ids[_mask])})

    return type_to_id, old_to_new

def encode_features(model, x):
    x_files, x_processes, x_sockets = 0, 0, 0
    files = x[(x[:, 0] == 0).nonzero().squeeze(1)]
    for h in [0, 1, 2, 3, 4]:
        x_files += model.embeddings[h](files[:, h])
    processes = x[(x[:, 0] == 1).nonzero().squeeze(1)]
    for h in [0, 5]:
        x_processes += model.embeddings[h](processes[:, h])
    sockets = x[(x[:, 0] == 2).nonzero().squeeze(1)]
    for h in [0, 6, 7, 8, 9]:
        x_sockets += model.embeddings[h](sockets[:, h])

    x_new = torch.zeros((x.shape[0], x_sockets.shape[-1])).to(x.device)
    x_new[(x[:, 0] == 0).nonzero().squeeze(1)] = x_files
    x_new[(x[:, 0] == 1).nonzero().squeeze(1)] = x_processes
    x_new[(x[:, 0] == 2).nonzero().squeeze(1)] = x_sockets
    return x_new


def encode_hetero(
    model, batch, msg, z, edge_index, last_update, t, n_id, src_indic: List[bool]
):
    types: List[str] = model.gnn[0].metadata[0]
    #time_start = time.time()
    n_type_to_idinbatch, globid_to_idinbatch = get_indices(batch.x[n_id, 0], n_id)
    #wandb.log({"get_indices": time.time() - time_start})
    time_start = time.time()
    z_new = {}
    last_update_new = {}
    # for each node_type
    for i, k in enumerate(types):
        if i in n_type_to_idinbatch:
            z_new[k] = z[n_type_to_idinbatch[i]]
            last_update_new[k] = last_update[n_type_to_idinbatch[i]]
        else:
            z_new[k] = z.new(0, z.shape[1])
            last_update_new[k] = last_update.new(0)
    
    msg_ind = msg.argmax(1).cpu()
    src_types, dst_types = batch.x[n_id[edge_index.view(-1)], 0].view(2, -1).cpu()
    src_and_dst = n_id[edge_index.view(-1)]  
    # map source and destinations from global ids to batch-wise ids
    src_and_dst = src_and_dst.cpu().apply_(globid_to_idinbatch.get)
    keys = [(types[src_types[i]], ("rel_" if src_indic[i] else "rev_rel_") + str(msg_ind[i].item()), types[dst_types[i]]) for i in range(edge_index.size(1))]

    # faster to do comparisons on strings rather than tuples, so we craft this string
    # made of "sourcetype.reltype.desttype"
    keys_str = [k[0]+"."+k[1]+"."+k[2] for k in keys]
    uqk = set(keys_str)  # find unique types in batch
    npkeys = np.array(keys_str)  # create np array for nonzero later

    # these are indexed by triples (sourcetype, reltype, desttype)
    edge_index_new: Dict[Tuple, Tensor] = {}
    msg_new: Dict[Tuple, Tensor] = {}
    t_new: Dict[Tuple, Tensor] = {}

    for k in uqk:
        # still use the actual tuple as the key rather than the string.
        kt = tuple(k.split("."))  
        # integer indexes in the batch that match the sought key triple (k)
        mask1: Tensor[int] = torch.tensor(np.nonzero((npkeys == k))[0], dtype=torch.long)
        # extend the mask so that we can use it with src_and_dst
        mask2 = mask1.clone()+len(npkeys)
        mask = torch.cat((mask1, mask2), 0)
        msg_new[kt] = torch.zeros((len(mask1), msg.size(1)))#.to(batch.x.device)
        edge_index_new[kt] = src_and_dst[mask].view(2, -1).clone().long()#.to(batch.x.device)  # [2, E]
        t_new[kt] = t[mask1]#.to(batch.x.device)
    msg_new = {k: v.to(batch.x.device, non_blocking=True) for k, v in msg_new.items()}
    edge_index_new = {k: v.to(batch.x.device, non_blocking=True) for k, v in edge_index_new.items()}
    t_new = {k: v.to(batch.x.device, non_blocking=True) for k, v in t_new.items()}

    #wandb.log({"edge_iteration": time.time() - time_start})
    # add empty arrays for all the triples (keys) we haven't encountered in the batch
    present_keys: Set[Tuple] = set([tuple(k.split(".")) for k in uqk])
    #time_start = time.time()
    missing_keys = set(model.gnn[0].metadata[1]) - present_keys
    for k in missing_keys:
        edge_index_new[k] = torch.zeros((2, 0)).view(2, 0).to(batch.x.device).long()
        msg_new[k] = torch.zeros((0, msg.size(1))).to(batch.x.device)
        t_new[k] = t.new(0).to(batch.x.device)
    #wandb.log({"edge_iteration_concat": time.time() - time_start})

    return (
        types,
        n_type_to_idinbatch,
        globid_to_idinbatch,
        edge_index_new,
        msg_new,
        last_update_new,
        t_new,
        z_new,
    )


class GenericModel(torch.nn.Module):
    
    def __init__(self, num_nodes, node_embedding_dim=[], memory=None, gnn=None, gnn_act=None, link_pred=None, include_features=True, edge_encoder=None, gnn_rev=None, atten_module=None, one_hot_dir=False):
        super(GenericModel, self).__init__()
        self.memory = memory
        self.gnn = gnn
        self.gnn_act = gnn_act
        self.link_pred = link_pred
        self.num_gnn_layers = 1
        self.num_nodes = num_nodes
        self.include_features = include_features
        if len(node_embedding_dim) > 0:
            if not include_features:
                node_embedding_dim = [node_embedding_dim[0]]
            self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(c, dim) for (c, dim) in node_embedding_dim])
        else:
            self.embeddings = []
        self.edge_encoder = edge_encoder
        self.gnn_rev = gnn_rev
        self.one_hot_dir = one_hot_dir
        self.atten_module = atten_module
        self.direction_encoder = torch.nn.Linear(1, 27)

    def reset_memory(self):
        if self.memory is not None: self.memory.reset_state()

    def update(self, src, pos_dst, t, msg, *args, **kwargs):
        if self.memory is not None: self.memory.update_state(src, pos_dst, t, msg)

    def detach_memory(self):
        if self.memory is not None: self.memory.detach()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if hasattr(self.memory, 'reset_parameters'):
            self.memory.reset_parameters()
        if hasattr(self.gnn, 'reset_parameters'):
                    self.gnn.reset_parameters()
        if hasattr(self.link_pred, 'reset_parameters'):
                    self.link_pred.reset_parameters()

    #def forward(self, batch, n_id, msg, t, edge_index, id_mapper, data, src_indic, neighbor_loader,  data_all, e_id):
    def forward(self, batch, n_id, msg, t, edge_index, id_mapper, src_indic):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if self.include_features:
                if len(self.embeddings) > 0:
                    x_new = encode_features(self, batch.x[n_id])
                else:
                    x_new = batch.x[n_id]
                z = torch.cat((z, x_new), dim=-1)
            else:
                if len(self.embeddings) > 0:
                    x = batch.x[n_id]
                    x_new = self.embeddings[0](x[:, 0])
                else:
                    x_new = batch.x[n_id]
                z = torch.cat((z, x_new), dim=-1)
        if self.encode_edge and not self.hetero_gnn:
            msg = self.edge_encoder(msg)
        if self.one_hot_dir:
            msg = msg + self.direction_encoder(src_indic.unsqueeze(1).float())
        if self.gnn is not None:
            #for i in range(edge_index.size(1)):
            #    assert (n_id[edge_index[0, i]].item() == data.src[i].item() and n_id[edge_index[1, i]].item() == data.dst[i].item()) or ( n_id[edge_index[0, i]].item() == data.dst[i].item() and n_id[edge_index[1, i]].item() == data.src[i].item())
            if self.hetero_gnn:
                #time0 = time.time()
                types, indices, old_to_new, edge_index,\
                msg, last_update, t, z = encode_hetero(self, batch, msg, z, edge_index, last_update, t, n_id, src_indic)
                #wandb.log({"hetero encoding time": time.time() - time0})

            time0 = time.time()
            
            if self.gnn_rev is not None:
                for gnn_layer, gnn_layer_rev in zip(self.gnn, self.gnn_rev):
                    z_in = gnn_layer(z, last_update, edge_index[:, src_indic], t[src_indic], msg[src_indic])
                    z_out = gnn_layer_rev(z, last_update, edge_index[:, ~src_indic], t[~src_indic], msg[~src_indic])
                    alpha = self.atten_module(torch.cat((z_in, z_out), dim=-1))
                    z = alpha*z_in + (1 - alpha)*z_out
                    z = self.gnn_act(z)
            else:
                for gnn_layer in self.gnn:
                    #time_layer = time.time()
                    z = gnn_layer(z, last_update, edge_index, t, msg)
                    if self.hetero_gnn:
                        #wandb.log({"number of edges": torch.sum(torch.tensor([msg[i].shape for i in msg]))})
                        #wandb.log({"number of edge types": ((torch.tensor([msg[i].shape for i in msg]))[:, 0] > 0 ).sum(0)})
                        #wandb.log({"gnn layer time": time.time() - time_layer})
                        for i, _ in enumerate(types):
                            if i in indices:
                                z[types[i]] = self.gnn_act(z[types[i]])
                    else:
                        z = self.gnn_act(z)
                if self.hetero_gnn:
                    out = z[list(z.keys())[0]].new(len(old_to_new), gnn_layer.out_dim)
                    for i, _ in enumerate(types):
                        if i in indices:
                            out[indices[i]] = z[types[i]]
                    z = out
                    #wandb.log({"number of nodes": z.shape[0]})
                    #wandb.log({"gnn time": time.time() - time0})
        target_message = self.edge_encoder(batch.msg) if self.include_edge else batch.msg
        pos_out = self.link_pred(z[id_mapper[src]], z[id_mapper[pos_dst]], target_message)
        neg_out = self.link_pred(z[id_mapper[src]], z[id_mapper[neg_dst]], target_message) if neg_dst is not None else None
        return pos_out, neg_out
    

class TGN(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int,
                 memory: bool = True, 
                 node_dim: int = 0, 
                 node_embedding_dim: List[Tuple]=[],
                 # GNN params
                 gnn_hidden_dim: List[int] = [],
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 include_edge: bool = False,
                 include_features: bool = True,
                 aggregator:str = 'last',
                 encode_edge:bool = False,
                 dir_GNN:bool = True,
                 hetero_gnn:bool = False,
                 data_metadata:Tuple = (),
                 log:bool = False,
                 hetero_transformer:bool=False, 
                 one_hot_dir:bool=False

        ):

        edge_encoder = torch.nn.Linear(edge_dim, edge_dim) if encode_edge or include_edge else None

        if aggregator == 'rnn':
            aggregator_module = AGGREGATOR_CONFS[aggregator](2*memory_dim + edge_dim + time_dim, 2*memory_dim + edge_dim + time_dim, log)
        else:
            aggregator_module = AGGREGATOR_CONFS[aggregator]()

        
        # Define memory
        if memory:
            memory = GeneralMemory(
                num_nodes,
                edge_dim,
                memory_dim,
                time_dim,
                message_module=IdentityMessage(edge_dim, memory_dim, time_dim, edge_encoder if encode_edge else None),
                aggregator_module=aggregator_module,
                rnn='GRUCell',
                init_time=init_time
            )
        else:
            memory = NoMemory(num_nodes, memory_dim, time_dim, init_time)


        # Define GNN
        gnn = torch.nn.Sequential()
        gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
        if len(node_embedding_dim) > 0:
            h_prev = memory_dim +  node_embedding_dim[0][1]
        else:
            h_prev = memory_dim +  node_dim

        for h in gnn_hidden_dim:
            if hetero_gnn:
                gnn.append(HeteroGraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                                mean_delta_t=mean_delta_t, std_delta_t=std_delta_t, metadata=data_metadata, hetero_transformer=hetero_transformer))
                
                """gnn.append(to_hetero(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                                mean_delta_t=mean_delta_t, std_delta_t=std_delta_t), data_metadata))"""
                
            else:
                gnn.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                                mean_delta_t=mean_delta_t, std_delta_t=std_delta_t))
            h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads

        if dir_GNN and not hetero_gnn:
            gnn_rev = torch.nn.Sequential()
            gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
            if len(node_embedding_dim) > 0:
                h_prev = memory_dim +  node_embedding_dim[0][1]
            else:
                h_prev = memory_dim +  node_dim

            for h in gnn_hidden_dim:
                gnn_rev.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                                    mean_delta_t=mean_delta_t, std_delta_t=std_delta_t))
                h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads
            atten_module = torch.nn.Linear(gnn_hidden_dim[-1]*2*2, 1)
        else:
            gnn_rev = None
            atten_module = None

        # Define the link predictor
        # NOTE: We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads
        link_pred = LinkPredictor(gnn_hidden_dim[-1] * 2, readout_hidden, out_channels=1, include_edge=include_edge, edge_dim=edge_dim)

        super().__init__(num_nodes, node_embedding_dim, memory, gnn, gnn_act, link_pred, include_features, edge_encoder, gnn_rev, atten_module, one_hot_dir)
        self.num_gnn_layers = len(gnn_hidden_dim)
        self.include_edge = include_edge
        self.encode_edge = encode_edge
        self.hetero_gnn = hetero_gnn
        