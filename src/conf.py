from utils import cartesian_product
from models import *


def get_TGN_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t):
    # Hetero-TGN: 'hetero_gnn' = [True], 'hetero_transformer' = [False]
    # HGT-TGN: 'hetero_gnn' = [True], 'hetero_transformer' = [True]
    # OHD-TGN: 'one_hot_dir' = [True]
    # DIR-TGN: 'dir_GNN' = [True]
    confs = {
        'aggregator': ['last', 'sum', 'mean'],
        'embedding_dim': [100, 64], #, 200, 64], #32, 64, 128], 
        'time_dim': [100], #32, 64, 128], 
        'lr': [0.00001, 0.000001], #, 0.001],#, 0.0001],
        'wd': [0.0001], #0.00001],
        'gnn_act': ['relu'],
        'sampler_size': [10], #10, 5],
        'include_edge': [False],
        'hetero_gnn':[False, True],
        'include_features': [True],
        'encode_edge': [True],
        'dir_GNN': [False, True],
        'memory':[True],
        'num_layers':[1, 2],
        'hetero_transformer':[True, False],
        'one_hot_dir':[True, False],
        'run_default_TGN':[False]
    }

    for params in cartesian_product(confs):
        if (params['hetero_gnn'] and params['dir_GNN']):
            continue
        if (params['one_hot_dir'] and params['dir_GNN']):
            continue
        if (params['one_hot_dir'] and params['hetero_gnn']):
            continue
        if (not params['hetero_gnn'] and params['hetero_transformer']):
            continue
        if (not params['run_default_TGN'] and not params['hetero_gnn'] and not params['dir_GNN'] and not params['one_hot_dir']):
            continue
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'node_embedding_dim':[(n, params['embedding_dim']) for n in node_num_embeddings],
                'memory': params['memory'],
                'memory_dim': params['embedding_dim'],
                'time_dim': params['time_dim'],
                'gnn_hidden_dim': [params['embedding_dim'] // 2]*params['num_layers'],
                'gnn_act': params['gnn_act'],
                'readout_hidden': max(1, params['embedding_dim']),
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time,
                'include_edge':params['include_edge'],
                'include_features':params['include_features'],
                'encode_edge': params['encode_edge'],
                'dir_GNN': params['dir_GNN'],
                'aggregator':params['aggregator'],
                'hetero_gnn':params['hetero_gnn'],
                'hetero_transformer': params['hetero_transformer'] if 'hetero_transformer' in params else False,
                'one_hot_dir': params['one_hot_dir']
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }




def get_Basic_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t):
    confs = {
        'embedding_dim':[0],
        'sampler_size': [1],
        'lr': [0.001], #0.001, 0.0001],
        'wd': [0.0001], #0.00001],
        'node_embedding_dim':[100],
        'include_edge': [False],
        'include_features': [False],
    }

    for params in cartesian_product(confs):
        yield { 
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd'],
            },
            'model_params':{
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'include_edge': params['include_edge'],
                'include_features':params['include_features'],
                'node_embedding_dim':[(n, params['node_embedding_dim']) for n in node_num_embeddings],
            },
            'sampler': {'size': params['sampler_size']},
            }

def get_GNN_conf(num_nodes, num_relations, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t):
    confs = {
        'num_layers': [1],
        'lr': [0.001],
        'wd': [0.0001],
        'sampler_size': [10],
        'directed': [False],
        'disjoint_train':[False],
        'node_embedding_dim': [100],
        'include_features':[True],
        'include_edge':[False],
    }

    for params in cartesian_product(confs):
        yield { 
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'model_params':{
                'num_layers': params['num_layers'],
                'node_embedding_dim':[(n, params['node_embedding_dim']) for n in node_num_embeddings],
                'num_relations': num_relations,
                'node_dim': node_dim,
                'in_channels': params['node_embedding_dim'],
                'hidden_channels': 32,
                'out_channels': params['node_embedding_dim'],
                'dropout': False,
                'include_features': params['include_features'],
                'include_edge': params['include_edge'],
            },
            'sampler': {'size': params['sampler_size'], 'directed': params['directed'], 'disjoint_train': params['disjoint_train']},
            }


_tgn_fun = lambda num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t: get_TGN_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t)
_basic_fun = lambda num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t: get_Basic_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t)
_gnn_fun = lambda num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t: get_GNN_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t)

tgn = 'TGN'
mlp, edgebank, gat, rgcn = 'MLP', 'EdgeBank', 'GAT', "RGCN"
MODEL_CONFS = {
    tgn: (TGN, _tgn_fun),
    mlp:(BaselineEdgePredictor, _basic_fun),
    edgebank:(EdgeBank, _basic_fun),
    gat:(GAT, _gnn_fun),
    rgcn:(RGCN, _gnn_fun),
}
