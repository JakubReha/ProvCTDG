import torch
from torch_geometric.data import Data
from torch import Tensor

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, confusion_matrix
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.sampler import (
    EdgeSamplerInput,
    NeighborSampler
)
from typing import Optional, Tuple
import numpy as np
import itertools
import random
import gc


AUC = 'auc'
F1_SCORE = 'f1_score'
ACCURACY = 'accuracy'
AVERAGE_PRECISION = 'ap'
MSE = 'mse'
MAE = 'mae'

CLASSIFICATION_SCORES = {
 AUC: roc_auc_score,
 F1_SCORE: f1_score,
 ACCURACY: accuracy_score,
 AVERAGE_PRECISION: average_precision_score
}

REGRESSION_SCORES = {
 MAE: torch.nn.L1Loss(),
 MSE: torch.nn.MSELoss(),
}

SCORE_NAMES = list(CLASSIFICATION_SCORES) + list(REGRESSION_SCORES)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_stats(data, split, init_time, ext_roll=False):
    if ext_roll:
        val_idx = int((data.ext_roll <= 0).sum())
        train_data = data[:val_idx]
    else:
        train_data, _, _ = data.train_val_test_split(val_ratio=split[0], test_ratio=split[1])

    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    last_timestamp = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    all_timediffs = []
    for src, dst, t in zip(train_data.src, train_data.dst, train_data.t):
        src, dst, t = src.item(), dst.item(), t.item()

        all_timediffs_src.append(t - last_timestamp_src.get(src, init_time))
        all_timediffs_dst.append(t - last_timestamp_dst.get(dst, init_time))
        all_timediffs.append(t - last_timestamp.get(src, init_time))
        all_timediffs.append(t - last_timestamp.get(dst, init_time))

        last_timestamp_src[src] = t
        last_timestamp_dst[dst] = t
        last_timestamp[src] = t
        last_timestamp[dst] = t
    assert len(all_timediffs_src) == train_data.num_events
    assert len(all_timediffs_dst) == train_data.num_events
    assert len(all_timediffs) == train_data.num_events * 2

    src_and_dst = all_timediffs_src + all_timediffs_dst
    all_timediffs_src = np.array(all_timediffs_src)
    all_timediffs_dst = np.array(all_timediffs_dst)
    all_timediffs = np.array(all_timediffs)
    mean_delta_t = np.mean(all_timediffs)
    std_delta_t = np.std(all_timediffs)

    print(f'avg delta_t(src): {np.mean(all_timediffs_src)} +/- {np.std(all_timediffs_src)}')
    print(f'avg delta_t(dst): {np.mean(all_timediffs_dst)} +/- {np.std(all_timediffs_dst)}')
    print(f'avg delta_t(src+dst): {np.mean(src_and_dst)} +/- {np.std(src_and_dst)}')
    print(f'avg delta_t(all): {mean_delta_t} +/- {std_delta_t}')

    del src_and_dst
    del all_timediffs_src
    del all_timediffs_dst
    del all_timediffs
    del train_data
    gc.collect()

    return mean_delta_t, std_delta_t


def optimizer_to(optim, device):
    # Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scoring(y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_confidence: torch.Tensor, 
            is_regression: bool = False, require_sigmoid: bool = True, labels: Optional[list] = None):
    s = {}
    if not is_regression:
        for k, func in CLASSIFICATION_SCORES.items():
            if k == AVERAGE_PRECISION or k == AUC:
                if require_sigmoid or k == AUC:
                    y_pred_confidence_ = y_pred_confidence.sigmoid()
                else:
                    y_pred_confidence_ = y_pred_confidence

                f = func(y_true, y_pred_confidence_)
            else:
                f = func(y_true, y_pred) #, average='weighted')
            s[k] = f
        s["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        s = {k: func(y_pred, y_true).detach().cpu().item() for k, func in REGRESSION_SCORES.items()}
    return s



def cartesian_product(params):
    # Given a dictionary where for each key is associated a lists of values, the function compute cartesian product
    # of all values. 
    # Example:
    #  Input:  params = {"n_layer": [1,2], "bias": [True, False] }
    #  Output: {"n_layer": [1], "bias": [True]}
    #          {"n_layer": [1], "bias": [False]}
    #          {"n_layer": [2], "bias": [True]}
    #          {"n_layer": [2], "bias": [False]}
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

ALL = 'all'
SPLIT = 'split'
dst_strategies = [SPLIT, ALL]
dst_strategies_help = (f'\n\t{ALL}: train, val, and test samplers always uses all the nodes in the data'
                       f'\n\t{SPLIT}: the train_sampler uses only the dst nodes in train set, val_sampler '
                       'uses train+val dst nodes, test_sampler uses all dst nodes in the data')
def get_node_sets(strategy, train_data, val_data, test_data):
    if strategy == ALL:
        src_node_set = torch.cat([train_data.src, val_data.src, test_data.src]).type(torch.long)
        dst_node_set = torch.cat([train_data.dst, val_data.dst, test_data.dst]).type(torch.long)
        train_src_nodes, train_dst_nodes = src_node_set, dst_node_set
        val_src_nodes, val_dst_nodes = src_node_set, dst_node_set
        test_src_nodes, test_dst_nodes = src_node_set, dst_node_set

    elif strategy == SPLIT:
        if hasattr(train_data, 'src'):
            train_src_nodes, train_dst_nodes = train_data.src.type(torch.long), train_data.dst.type(torch.long)
            val_src_nodes, val_dst_nodes = (
                torch.cat([train_data.src, val_data.src]).type(torch.long),
                torch.cat([train_data.dst, val_data.dst]).type(torch.long)
            )
            test_src_nodes, test_dst_nodes = (
                torch.cat([train_data.src, val_data.src, test_data.src]).type(torch.long),
                torch.cat([train_data.dst, val_data.dst, test_data.dst]).type(torch.long)
            )
        else:
            train_src_nodes, train_dst_nodes = train_data.edge_index[0].type(torch.long), train_data.edge_index[1].type(torch.long)
            val_src_nodes, val_dst_nodes = (
                torch.cat([train_data.edge_index[0], val_data.edge_index[0]]).type(torch.long),
                torch.cat([train_data.edge_index[1], val_data.edge_index[1]]).type(torch.long)
            )
            test_src_nodes, test_dst_nodes = (
                torch.cat([train_data.edge_index[0], val_data.edge_index[0], test_data.edge_index[0]]).type(torch.long),
                torch.cat([train_data.edge_index[1], val_data.edge_index[1], test_data.edge_index[1]]).type(torch.long)
            )
    else:
        raise NotImplementedError()
    
    return train_src_nodes, train_dst_nodes, val_src_nodes, val_dst_nodes, test_src_nodes, test_dst_nodes


def merge_static_data(data1, data2):
    data = Data(x=data1.x,
                edge_index=torch.cat((data1.edge_index, data2.edge_index), dim=1),
                edge_attr=torch.cat((data1.edge_attr, data2.edge_attr)),
                #ext_roll=torch.cat((data1.ext_roll, data2.ext_roll)),
                hash_id=torch.cat((data1.hash_id, data2.hash_id)),
                malicious=torch.cat((data1.malicious, data2.malicious)))
    return data


def to_undirected(data):
    data.edge_index = torch.cat((data.edge_index, torch.stack((data.edge_index[1], data.edge_index[0]))), dim=1)
    data.edge_attr =  torch.cat((data.edge_attr, data.edge_attr.clone() + 27))
    data.hash_id = torch.cat((data.hash_id, data.hash_id))
    data.malicious = torch.cat((data.malicious, data.malicious))
    return data


class StaticNeighborLoader:
        def __init__(self, train_mess_data, train_data, val_data, num_nodes, subgraph_type=True):
            self.train_mess_data = train_mess_data
            self.train_data = train_data
            self.val_data = val_data
            
            self.train_sampler = NeighborSampler( train_mess_data, num_neighbors=num_nodes, directed=subgraph_type)
            self.val_sampler = NeighborSampler( train_data, num_neighbors=num_nodes, directed=subgraph_type)
            self.test_sampler = NeighborSampler( val_data, num_neighbors=num_nodes, directed=subgraph_type)

        def __call__(self, edge_index, neg_dst, mode):
            input_data = EdgeSamplerInput(input_id=torch.arange(len(edge_index[0])) , row=torch.cat((edge_index[0].clone(), edge_index[0].clone())), col=torch.cat((edge_index[1].clone(), neg_dst.cpu())))
            if 'train' in mode:
                data = self.train_mess_data
                out = self.train_sampler.sample_from_edges(input_data, neg_sampling=None)
            elif 'val' in mode:
                data = self.train_data
                out = self.val_sampler.sample_from_edges(input_data, neg_sampling=None)
            elif 'test' in mode:
                data = self.val_data
                out = self.test_sampler.sample_from_edges(input_data, neg_sampling=None)
            
            src = out.metadata[1][0][:len(edge_index[0])]
            pos_dst = out.metadata[1][1][:len(edge_index[0])]
            neg_dst = out.metadata[1][1][len(edge_index[0]):]
            edge_index = torch.stack((out.row, out.col))

            return src, pos_dst, neg_dst, edge_index, data.edge_attr[out.edge.long()], data.x[out.node.long()]
            

        def reset_state(self):
            return None
        
        def insert(self, a, b):
            return None
        

class LinkStaticLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index, msg, hash_id, malicious):
        self.edge_index = edge_index
        self.msg = msg.T
        self.hash_id = hash_id
        self.malicious = malicious

    def __len__(self):
        return len(self.edge_index[0])

    def __getitem__(self, idx):
        return {"edge_index": self.edge_index[:, idx], "hash_id":self.hash_id[idx], "malicious":self.malicious[idx], "msg":self.msg[:, idx]}


class LinkStaticLoader(torch.utils.data.DataLoader):
    def __init__(self, edge_index, msg, hash_id, malicious, batch_size: int = 1):
        super().__init__(LinkStaticLoaderDataset(edge_index, msg, hash_id, malicious), batch_size, shuffle=True)


def get_indices_old(index):
    out = dict()
    for i in range(len(index)):
        key = index[i].item()
        if key in out:
            out[key].append(i)
        else:
            out[key] = [i]
    return out

def get_indices(ntypes: Tensor, msg):
    type_to_id = dict()
    if len(ntypes)==0:
        available_ntypes = []
    else:
        available_ntypes=range(ntypes.max()+1)
    msgs = []
    for _type in available_ntypes:
        # integer indexes of nodes that have type `_type`
        _mask: Tensor[int] = torch.nonzero(ntypes==_type)
        type_to_id[_type] = _mask.view(-1).tolist()
        msgs.append(msg[type_to_id[_type]])
    return type_to_id, msgs


class LastNeighborLoader(object):
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long,
                                     device=device)
        self.src_indicator = torch.empty((num_nodes, size), dtype=torch.bool,
                                     device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long,
                                device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        src_indicator = self.src_indicator[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]
        src_indicator = src_indicator[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id, src_indicator

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        src_indicator = torch.cat([torch.ones_like(src).bool(), torch.zeros_like(dst).bool()], dim=0).bool()
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]
        src_indicator = src_indicator[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        dense_src_indicator = e_id.new_empty(n_id.numel() * self.size).bool()
        dense_src_indicator[dense_id] = src_indicator
        dense_src_indicator = dense_src_indicator.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)
        src_indicator = torch.cat(
            [self.src_indicator[n_id, :self.size], dense_src_indicator], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)
        self.src_indicator[n_id] = torch.gather(src_indicator, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)
