from utils import (set_seed, SCORE_NAMES, dst_strategies, dst_strategies_help, 
                   REGRESSION_SCORES, CLASSIFICATION_SCORES, compute_stats)
from train_link import link_prediction, link_prediction_single
from utils import set_seed, SCORE_NAMES, dst_strategies, dst_strategies_help, REGRESSION_SCORES, CLASSIFICATION_SCORES
from negative_sampler import neg_sampler_names
from datasets import get_dataset, DATA_NAMES, DARPADataset_Static
from conf import MODEL_CONFS
import pandas as pd
import warnings
import argparse
import datetime
import pickle
import time
import tqdm
import ray
import os
import gc
import subprocess

def compute_row(test_score, val_score, train_score, best_epoch, res_conf):
    row = {}
    for label, score_dict in [('test', test_score), ('val', val_score), ('train', train_score)]:
        for strategy in score_dict.keys(): 
            for k, v in score_dict[strategy].items():
                row[f'{label}_{strategy}_{k}'] = v

    for k in res_conf.keys():
        if isinstance(res_conf[k], dict):
            for sk in res_conf[k]:
                row[f'{k}_{sk}'] = res_conf[k][sk]
        else:
            row[k] = res_conf[k]
    row.update({f'best_epoch': best_epoch})
    return row


if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='The path to the directory where data files are stored.', default='./DATA/')
    parser.add_argument('--data_name', help='The data name.', default=DATA_NAMES[0], choices=DATA_NAMES)
    parser.add_argument('--version', help='Version of DARPA dataset', default='temporal', choices=['temporal', 'static'])
    parser.add_argument('--save_dir', help='The path to the directory where checkpoints/results are stored.', default='./TGN_debug/')
    parser.add_argument('--model', help='The model name.', default=list(MODEL_CONFS)[0], choices=MODEL_CONFS.keys())
    parser.add_argument('--neg_sampler', help='The negative_sampler name.', default=neg_sampler_names[1], choices=neg_sampler_names)
    parser.add_argument('--strategy', help=f'The strategy to sample train, val, and test sets of dst nodes used by the negative_sampler.{dst_strategies_help}', default=dst_strategies[0], choices=dst_strategies)
    parser.add_argument('--use_all_strategies_eval', help='Use all strategies during the final evaluation.', action="store_true")
    parser.add_argument('--no_check_link_existence', help=f'The negative sampler does not check if the sampled negative link exists in the graph during sampling.', action='store_true')
    parser.add_argument('--no_normalize_delta_t', help=f'Do not normalize the time difference between current t and last update.', action='store_true')
    parser.add_argument('--link_regression', help='Instead of link prediction run a link regression task.', action='store_true')
    parser.add_argument('--num_runs', help='The number of random initialization per conf.', default=5, type=int)
    parser.add_argument('--split', help='(val_ratio, test_ratio) split ratios.', nargs=2, default=[.15, .15])
    parser.add_argument('--epochs', help='The number of epochs.', default=5, type=int)
    parser.add_argument('--batch', help='The batch_size.', default=256, type=int)
    parser.add_argument('--patience', help='The early stopping patience, ie train is stopped if no score improvement after X epochs.', default=50, type=int)
    parser.add_argument('--exp_seed', help='The experimental seed.', default=9, type=int)
    parser.add_argument('--metric', help='The optimized metric.', default=list(SCORE_NAMES)[3], choices=list(SCORE_NAMES))
    parser.add_argument('--debug', help='Debug mode.', action='store_true')
    parser.add_argument('--cpu', help='cpu mode.', action='store_true')
    parser.add_argument('--return_predictions', help='return predictions', action='store_true')
    parser.add_argument('--own_train_test_split', help='own train test split', action='store_true')
    parser.add_argument('--wandb', help='Compute Weights and Biases log.', action='store_true')
    parser.add_argument('--cluster', help='Experiments run on a cluster.', action='store_true')
    parser.add_argument('--parallelism', help='The degree of parallelism, ie, maximum number of parallel jobs.', default=None, type=int)
    parser.add_argument('--overwrite_ckpt', help='Overwrite checkpoint.', action='store_true')
    parser.add_argument('--verbose', help='Every <patience> epochs it prints the average time to compute an epoch.', action='store_true')
    parser.add_argument('--inference', help='Run inference only', action='store_true')

    parser.add_argument('--reset_memory_eval', help='Reset memory before every evaluation (val/test).', action='store_true')
    
    
    args = parser.parse_args()
        
    if 'darpa' in args.data_name: args.return_predictions = True
        
    assert not (args.link_regression and args.use_all_dst_strategies_eval), 'Link regression does not require neg sampling strategies'
    assert args.link_regression == (args.metric in REGRESSION_SCORES), 'Link regression requires regression metrics'
    assert args.link_regression != (args.metric in CLASSIFICATION_SCORES), 'Link prediction requires classification metrics'

    set_seed(args.exp_seed)

    if not args.debug:
        if args.cluster:
            # Kubernetes cluster init
            runtime_env = {
                "working_dir": os.getcwd(), # working_dir is the directory that contains main.py
            }

            # Get head name
            cmd = ("microk8s kubectl get pods --selector=ray.io/node-type=head -o "
                   "custom-columns=POD:metadata.name --no-headers").split(" ")
            head_name = subprocess.check_output(cmd).decode("utf-8").strip()

            # Get head ip
            cmd = ("microk8s kubectl get pod " + head_name + " --template '{{.status.podIP}}'").split(" ")
            head_ip = subprocess.check_output(cmd).decode("utf-8").strip().replace("'", "")

            ray.init(f"ray://{head_ip}:10001", runtime_env=runtime_env)


            # SLURM cluster init
            # ray.init(address=os.environ.get("ip_head"), 
            #         _redis_password=os.environ.get("redis_password"))  # ray initialization on a cluster
            print(f"Resources: cluster")
        else:
            ray.init(num_cpus=int(os.environ.get('NUM_CPUS', 2)), 
                     num_gpus=int(os.environ.get('NUM_GPUS', 0)))
            print(f"Resources: CPUS: {os.environ.get('NUM_CPUS', 2)}, GPUS={os.environ.get('NUM_GPUS', 0)}")

    args.save_dir = os.path.abspath(args.save_dir)
    args.data_dir = os.path.join(os.path.abspath(args.data_dir))
    if not os.path.isdir(args.data_dir): os.makedirs(args.data_dir)
    
    result_path = os.path.join(args.save_dir, args.model)
    if not os.path.isdir(result_path): os.makedirs(result_path)

    ckpt_path = os.path.join(result_path, 'ckpt')
    if not os.path.isdir(ckpt_path): os.makedirs(ckpt_path)

    print(f'\n{args}\n')
    print(f'Data dir: {args.data_dir}')
    print(f'Results dir: {result_path}')
    print(f'Checkpoints dir: {ckpt_path}\n')

    partial_res_pkl = os.path.join(result_path, 'partial_results.pkl')
    partial_res_csv = os.path.join(result_path, 'partial_results.csv')
    final_res_csv = os.path.join(result_path, 'model_selection_results.csv')

    data, _ = get_dataset(root=args.data_dir, name=args.data_name, version=args.version, seed=args.exp_seed)

    if isinstance(data, DARPADataset_Static):
        num_nodes, edge_dim = data.num_nodes, data.msg.max() + 1  
        node_dim = data.x.shape[-1] if hasattr(data, 'x') else 0
        init_time = 0
        node_num_embeddings = data.x.max(dim=0).values+1 if 'darpa' in args.data_name else []
    else:
        num_nodes, edge_dim = data.num_nodes, data.msg.shape[-1] 
        node_dim = data.x.shape[-1] if hasattr(data, 'x') else 0
        node_num_embeddings = data.x.max(dim=0).values+1 if 'darpa' in args.data_name else []
        
        init_time = data.t[0] if hasattr(data, 't') else 0

    if args.no_normalize_delta_t or "static" in args.version:
        mean_delta_t, std_delta_t = 0., 1.
    else:
        stat_path = os.path.join(args.data_dir, args.data_name.lower(), 'delta_t_stats.pkl')
        if os.path.exists(stat_path):
            mean_delta_t, std_delta_t = pickle.load(open(stat_path, 'rb')) 
        else:
            mean_delta_t, std_delta_t = compute_stats(data, args.split, init_time, ext_roll='darpa' in args.data_name)
            pickle.dump((mean_delta_t, std_delta_t), open(stat_path, 'wb'))
    del data
    gc.collect()

    model_instance, get_conf = MODEL_CONFS[args.model]

    num_conf = len(list(get_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t)))
    pbar = tqdm.tqdm(total= num_conf*args.num_runs)
    df = []
    ray_ids = []
    for conf_id, conf in enumerate(get_conf(num_nodes, edge_dim, node_dim, node_num_embeddings, init_time, mean_delta_t, std_delta_t)):
        for i in range(args.num_runs):
            conf.update({
                'conf_id': conf_id,
                'seed': i,
                'result_path': result_path,
                'ckpt_path': ckpt_path,
            })
            conf.update(vars(args))

            if args.debug:
                    test_score, val_score, train_score, best_epoch, res_conf = link_prediction_single(model_instance, conf)
                    df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
                    pickle.dump(df, open(partial_res_pkl, 'wb'))
                    pbar.update(1)
            else:
                if args.cluster:
                    if 'trace' in args.data_name:
                        memory = 30000 * 1024 * 1024
                        num_cpus = 2
                    else:
                        memory = 15000 * 1024 * 1024
                        num_cpus = 2
                    ray_ids.append(link_prediction.options(memory=memory, num_cpus=num_cpus, num_gpus=0).remote(model_instance, conf))
                else:    
                    ray_ids.append(link_prediction.remote(model_instance, conf))
            
            if args.parallelism is not None:
                while len(ray_ids) > args.parallelism:
                    done_id, ray_ids = ray.wait(ray_ids)
                    test_score, val_score, train_score, best_epoch, res_conf = ray.get(done_id[0])
                    if test_score is None:
                        pbar.update(1)
                        continue
                    df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
                    pickle.dump(df, open(partial_res_pkl, 'wb'))
                    pd.DataFrame(df).to_csv(partial_res_csv)
                    pbar.update(1)
                    gc.collect()
            gc.collect()
    
    while len(ray_ids):
        done_id, ray_ids = ray.wait(ray_ids)
        test_score, val_score, train_score, best_epoch, res_conf = ray.get(done_id[0])
        if test_score is None:
            continue
        df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
        pickle.dump(df, open(partial_res_pkl, 'wb'))
        pd.DataFrame(df).to_csv(partial_res_csv)
        pbar.update(1)
        gc.collect()

    df = pd.DataFrame(df)

    # Aggregate results over multiple runs
    # and sort them by best val score
    aggregated_df = []
    for conf_id, gdf in df.groupby('conf_id'):
        if args.num_runs == 1:
            row = gdf.iloc[0]
        else:
            row = {}
            for k in gdf.columns:
                if k == 'seed': 
                    row[k] = gdf[k].values 
                if 'test' in k or 'val' in k or 'train' in k or k == 'best_epoch':
                    row[f'{k}_mean'] = gdf[k].values.mean() if 'confusion_matrix' in k else gdf[k].mean()
                    row[f'{k}_std'] = gdf[k].values.std() if 'confusion_matrix' in k else gdf[k].std()
                else:
                    row[k] = gdf.iloc[0][k]
        aggregated_df.append(row)
    aggregated_df = pd.DataFrame(aggregated_df)
    aggregated_df = aggregated_df.sort_values(f'val_{args.strategy}_{args.metric}_mean' if args.num_runs > 1 else f'val_{args.strategy}_{args.metric}', 
                                              ascending=args.link_regression)
    aggregated_df.to_csv(final_res_csv)
    print(aggregated_df.iloc[0].to_string())

    t1 = time.time()
    print(f'Main ended in {datetime.timedelta(seconds=t1 - t0)}')
