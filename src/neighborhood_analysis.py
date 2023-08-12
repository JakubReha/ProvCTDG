import torch
from utils import get_node_sets, scoring, optimizer_to, set_seed, dst_strategies, REGRESSION_SCORES, merge_static_data, StaticNeighborLoader, LinkStaticLoader, to_undirected, LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from datasets import get_dataset
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import pdb
import collections


def link_prediction_single(conf):

    # Set the configuration seed
    set_seed(conf['seed'])
    if not conf['cpu']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    data, _ = get_dataset(root=conf['data_dir'], name=conf['data_name'], version=conf['version'], seed=conf['exp_seed'], metadata= conf['model_params']['hetero_gnn'] if 'hetero_gnn' in conf['model_params'] else False )
    
    data = data.to(device)
    if 'darpa' in conf['data_name']:
        val_idx = int((data.ext_roll <= 0).sum())
        test_idx = int((data.ext_roll <= 1).sum())
        train_data, val_data, test_data = data[:val_idx], data[val_idx:test_idx], data[test_idx:]
    else:
        train_data, val_data, test_data = data.train_val_test_split(val_ratio=conf['split'][0], test_ratio=conf['split'][1])

    train_loader = TemporalDataLoader(train_data, batch_size=conf['batch'])
    val_loader = TemporalDataLoader(val_data, batch_size=conf['batch'])
    test_loader = TemporalDataLoader(test_data, batch_size=conf['batch'])

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=conf['sampler']['size'], device='cpu')
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    neighbor_loader.reset_state()

    out_train = eval(phase='train', data=data, loader=train_loader, 
        neighbor_loader=neighbor_loader,
        helper=assoc, device=device)
    
    eval(phase='val', data=data, loader=val_loader, 
        neighbor_loader=neighbor_loader,
        helper=assoc, device=device)

    out_test = eval(phase='test', data=data, loader=test_loader,
        neighbor_loader=neighbor_loader,
        helper=assoc, device=device)
    return out_train, out_test




@torch.no_grad()
def eval(phase, data, loader, neighbor_loader, helper, device='cpu'):
    t0 = time.time()
    out = None
    if phase == 'test' or phase == 'train':
        out = {}

    y_pred_list, y_true_list, y_pred_confidence_list, hash_id_list, malicious_list = [], [], [], [], []
    for batch in tqdm(loader):
        batch = batch.to(device)
        src, pos_dst, t, msg, static_x = batch.src, batch.dst, batch.t, batch.msg, batch.x

        if phase == 'test' or phase == 'train':
            original_n_id = torch.cat([src, pos_dst])
        else:
            original_n_id = torch.cat([src, pos_dst]).unique()

        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[original_n_id.to(neighbor_loader.e_id.device)]
        src_indic = torch.ones_like(t).bool()
        if phase == 'test' or phase == 'train':
            for i, node in enumerate(original_n_id):
                if i >= len(src):
                    pos = 'dst'
                else:
                    pos = 'src'
                i = i % len(src)
                if batch.hash_id[i].item() not in out:
                    out[batch.hash_id[i].item()] = {}
                #elif 'dst' in out[batch.hash_id[i].item()]:
                #    pdb.set_trace()
                n_id, edge_index, e_id, src_indic = neighbor_loader(node.unsqueeze(0).to(neighbor_loader.e_id.device))
                out[batch.hash_id[i].item()][pos] = list(set(torch.unique(n_id).tolist()) - set([node.item()]))

        else:
            n_id, edge_index, e_id, src_indic = neighbor_loader(original_n_id.to(neighbor_loader.e_id.device))
        n_id, edge_index, e_id, src_indic = n_id.to(device), edge_index.to(device), e_id.to(device), src_indic.to(device)
            
            
        if hasattr(batch, "hash_id") or (isinstance(batch, dict) and "hash_id" in batch):
            hash_id_list.append(batch['hash_id'])
            #positive samples
            malicious_list.append(batch['malicious'])

        # Update memory and neighbor loader with ground-truth state.
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())

    return out
                                        
if __name__ == '__main__':
    conf = {'sampler': {'size': 10, 'directed': False, 'disjoint_train': False},
            'conf_id': 0, 'seed': 0, 'data_name': 'darpa_trace_0to210', 'version': 'temporal',
            'strategy': 'split', 'use_all_strategies_eval': False, 'no_check_link_existence': True, 'no_normalize_delta_t': False,
            'link_regression': False, 'num_runs': 1, 'split': [0.15, 0.15], 'epochs': 1, 'batch': 200, 'exp_seed': 9,
            'metric': 'ap', 'debug': True, 'cpu': True, 'own_train_test_split': False,
            'parallelism': 1, 'overwrite_ckpt': True, 'verbose': False, 'inference': False, 'reset_memory_eval': False,
            'model_params': {}, 'data_dir': '/home/jreha/data/parsed_for_training'
                }
    if not os.path.isfile('10_neighbours_new.p'):
        out_train, out_test = link_prediction_single(conf)
        with open('10_neighbours_new_test.p', 'wb') as fp:
            pickle.dump(out_test, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open('10_neighbours_new_train.p', 'wb') as fp:
            pickle.dump(out_train, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('10_neighbours_new_test.p', 'rb') as fp:
            out_test = pickle.load(fp)
        with open('10_neighbours_new_train.p', 'rb') as fp:
            out_train = pickle.load(fp)

    # load nodes
    prediction_folder = '/mnt/vdc/kuba/RESULTS/TGN_memory3_experiment_trace_0to210/TGN/ckpt/'
    model_name = 'TGN_without_memory_trace'
    ground_truth_path = '/home/jreha/jakub-reha/darpa_labelling/groundtruth/'
    conf_id = '0'
    dataset = 'TRACE'
    dataset_path = '/home/jreha/data/parsed_for_training/darpa_trace_0to210'
    nodes = pd.read_csv(os.path.join(dataset_path, "attributed_nodes.csv"),  index_col=0)
    edges = pd.read_csv(os.path.join(dataset_path, "edges.csv"))

    if dataset == 'TRACE':
        split = "0to210"
    elif dataset == 'THEIA':
        split = "0to25"
    else:
        raise NotImplemented
    
    src = nodes.iloc[edges.src.values]
    dst = nodes.iloc[edges.dst.values]
    for k in src:
        edges[k+"_src"] = src[k].values

    for k in dst:
        edges[k+"_dst"] = dst[k].values
    

    n_seeds = 5
    aggregated_predictions = 0
    for seed in range(0, n_seeds):
        prediction_path = os.path.join(prediction_folder, f"split_conf_{conf_id}_detection_results-{split}_seed_{seed}.csv")
        predictions = pd.read_csv(prediction_path)
        aggregated_predictions += predictions.prob.values
    aggregated_predictions /= n_seeds
    mask = ~aggregated_predictions.round().astype(bool)
    test_set = edges[edges.ext_roll == 2]
    fp = test_set[mask & (~test_set.malicious)]
    tn = test_set[~mask & (~test_set.malicious)]
    fp_src = test_set[mask & (~test_set.malicious) & (test_set['exe_src']=='firefox')]
    fp_dst = test_set[mask & (~test_set.malicious) & (test_set['exe_dst']=='firefox')]

    firefox_src_fp_hashes = fp_src.hash_id
    firefox_dst_fp_hashes = fp_dst.hash_id
    fp_src_out = {}
    fp_dst_out = {}
    fp_out = {}
    tn_out = {}


    for edge in fp.hash_id:
        fp_out[edge] = out_test[edge]
    for edge in tn.hash_id:
        tn_out[edge] = out_test[edge]

    tn_hist = [len(tn_out[i]['src']) for i in tn_out if 'src' in tn_out[i]] + [len(tn_out[i]['dst']) for i in tn_out if 'dst' in tn_out[i]]
    fp_hist = [len(fp_out[i]['src']) for i in fp_out if 'src' in fp_out[i]] + [len(fp_out[i]['dst']) for i in fp_out if 'dst' in fp_out[i]]
    train_hist = [len(out_train[i]['src']) for i in out_train if 'src' in out_train[i]] + [len(out_train[i]['dst']) for i in out_train if 'dst' in out_train[i]]
    print(len(tn_hist))
    print(len(fp_hist))

    bins = np.arange(12) - 0.5
    plt.figure()
    # plt.figure(figsize=(6.4,4.8*1.4))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams['axes.axisbelow'] = True
    plt.hist([train_hist, tn_hist, fp_hist], bins, label=['nodes in train', 'nodes in TN', 'nodes in FP'], weights=[np.ones(len(train_hist))/len(train_hist), np.ones(len(tn_hist))/len(tn_hist), np.ones(len(fp_hist))/len(fp_hist)], color=['#ffb000', '#fe6100', '#648fff'])
    step = 1
    plt.grid(axis='y')
    plt.xticks(np.arange(0, 10 + step, step))
    plt.xlabel("Temporal neighbourhood size")
    plt.ylabel("Normalized counts")

    plt.legend()
    plt.savefig("tn_vs_fp_vs_train_grid.pdf", bbox_inches='tight')

    """src_src = [len(fp_src_out[i]['src']) for i in fp_src_out if 'src' in fp_src_out[i]]
    src_dst = [len(fp_src_out[i]['dst']) for i in fp_src_out if 'dst' in fp_src_out[i]]
    dst_src = [len(fp_dst_out[i]['src']) for i in fp_dst_out if 'src' in fp_dst_out[i]]
    dst_dst = [len(fp_dst_out[i]['dst']) for i in fp_dst_out if 'dst' in fp_dst_out[i]]
    
    arr = np.concatenate([fp.src, fp.dst]).astype(str)
    print(collections.Counter(arr).most_common(10))
    plt.figure()
    plt.hist(arr)
    plt.savefig('fp_firefox_hist.png')
    plt.figure()
    plt.hist(src_src)
    plt.savefig('src_src.png')
    plt.figure()
    plt.hist(src_dst)
    plt.savefig('src_dst.png')
    plt.figure()
    plt.hist(dst_src)
    plt.savefig('dst_src.png')
    plt.figure()
    plt.hist(dst_dst)
    plt.savefig('dst_dst.png')"""




    print("average number of neighbours when firefox is source")
    print(f"src: {np.mean([len(fp_src_out[i]['src']) for i in fp_src_out if 'src' in fp_src_out[i]])}+-{np.std([len(fp_src_out[i]['src']) for i in fp_src_out if 'src' in fp_src_out[i]])}")
    print(f"dst: {np.mean([len(fp_src_out[i]['dst']) for i in fp_src_out if 'dst' in fp_src_out[i]])}+-{np.std([len(fp_src_out[i]['dst']) for i in fp_src_out if 'dst' in fp_src_out[i]])}")

    print("average number of neighbours when firefox is destination")
    print(f"src: {np.mean([len(fp_dst_out[i]['src']) for i in fp_dst_out if 'src' in fp_dst_out[i]])}+-{np.std([len(fp_dst_out[i]['src']) for i in fp_dst_out if 'src' in fp_dst_out[i]])}")
    print(f"dst: {np.mean([len(fp_dst_out[i]['dst']) for i in fp_dst_out if 'dst' in fp_dst_out[i]])}+-{np.std([len(fp_dst_out[i]['dst']) for i in fp_dst_out if 'dst' in fp_dst_out[i]])}")
    print("done")
