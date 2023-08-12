import torch

from utils import get_node_sets, scoring, optimizer_to, set_seed, dst_strategies, REGRESSION_SCORES, merge_static_data, StaticNeighborLoader, LinkStaticLoader, to_undirected, LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from datasets import get_dataset
import negative_sampler
import numpy as np
import pandas as pd
import datetime
import wandb
import time
import ray
import os


def train(data, model, optimizer, train_loader, criterion, neighbor_loader, helper, train_neg_sampler=None, device='cpu', backward=True, static=False, conf=None):
    model.train()
    # Start with a fresh memory and an empty graph
    model.reset_memory()
    neighbor_loader.reset_state()

    for batch in train_loader:
        t_batch_start = time.time()
        optimizer.zero_grad()
        if not static:
            batch = batch.to(device)
            src, pos_dst, t, msg, static_x = batch.src, batch.dst, batch.t, batch.msg, batch.x

            if train_neg_sampler is None:
                # NOTE: When the train_neg_sampler is None we are doing link regression
                original_n_id = torch.cat([src, pos_dst]).unique()
            else:
                t_start = time.time()
                # Sample negative destination nodes.
                neg_dst = train_neg_sampler.sample(src, batch.x[batch.dst, 0]).to(device)
                original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
                batch.neg_dst = neg_dst
                if conf['debug']:
                    if conf['wandb']:
                        wandb.log({"negative sampling time": time.time() - t_start})

            n_id = original_n_id
            edge_index = torch.empty(size=(2,0)).long()
            t_start = time.time()
            e_id = neighbor_loader.e_id[n_id.to(neighbor_loader.e_id.device)]
            src_indic = torch.ones_like(t).bool()
            for _ in range(model.num_gnn_layers):
                n_id, edge_index, e_id, src_indic = neighbor_loader(n_id.to(neighbor_loader.e_id.device))
            n_id, edge_index, e_id, src_indic = n_id.to(device), edge_index.to(device), e_id.to(device), src_indic.to(device)
            if conf['debug']:
                if conf['wandb']:
                    wandb.log({"neighbor loader time": time.time() - t_start})
            helper[n_id] = torch.arange(n_id.size(0), device=device)
            
            pos_out, neg_out = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device), t=data.t[e_id].to(device),
                                    edge_index=edge_index, id_mapper=helper, src_indic=src_indic)
        else:
            t = None
            static_x = None
            neg_dst = train_neg_sampler.sample(batch['edge_index'][:, 0], data.x[batch['edge_index'][:, 1].long(), 0]).to(device)
            src, pos_dst, neg_dst, edge_index, msg, x = neighbor_loader(batch['edge_index'].T, neg_dst, 'train')
            pos_out, neg_out = model(batch=batch, src=src.to(device), pos_dst=pos_dst.to(device), neg_dst=neg_dst.to(device),
                                     msg=msg.to(device), x=x.to(device), edge_index=edge_index.to(device))

        if train_neg_sampler is None:
            loss = criterion(pos_out, batch.y)
        else:
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, static_x)
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())
        
        if backward:
            t_backwards_start = time.time()
            loss.backward()
            optimizer.step()
            model.detach_memory()
        if conf['debug'] and conf['wandb']:
                wandb.log({"time per backwards": time.time() - t_backwards_start})
                wandb.log({"time per batch": time.time() - t_batch_start})

@torch.no_grad()
def eval(data, model, loader, criterion, neighbor_loader, helper, neg_sampler=None, eval_seed=12345,
         return_predictions=False, device='cpu', eval_name='eval', wandb_log=False, static=False):
    t0 = time.time()
    model.eval()

    y_pred_list, y_true_list, y_pred_confidence_list, hash_id_list, malicious_list = [], [], [], [], []
    for batch in loader:
        if not static:
            batch = batch.to(device)
            src, pos_dst, t, msg, static_x = batch.src, batch.dst, batch.t, batch.msg, batch.x

            if neg_sampler is None:
                # NOTE: When the neg_sampler is None we are doing link regression
                original_n_id = torch.cat([src, pos_dst]).unique()
            else:
                # Sample negative destination nodes
                neg_dst = neg_sampler.sample(src, batch.x[batch.dst, 0], eval=True, eval_seed=eval_seed).to(device) # Ensure deterministic sampling across epochs
                original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
                batch.neg_dst = neg_dst

            n_id = original_n_id
            edge_index = torch.empty(size=(2,0)).long()
            e_id = neighbor_loader.e_id[n_id.to(neighbor_loader.e_id.device)]
            src_indic = torch.ones_like(t).bool()
            for _ in range(model.num_gnn_layers):
                n_id, edge_index, e_id, src_indic = neighbor_loader(n_id.to(neighbor_loader.e_id.device))
            n_id, edge_index, e_id, src_indic = n_id.to(device), edge_index.to(device), e_id.to(device), src_indic.to(device)

            helper[n_id] = torch.arange(n_id.size(0), device=device)
            
            pos_out, neg_out = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device), t=data.t[e_id].to(device),
                                    edge_index=edge_index, id_mapper=helper, src_indic=src_indic)
        else:
            t = None
            static_x = None
            neg_dst = neg_sampler.sample(batch['edge_index'].T[0], data.x[batch['edge_index'][:, 1].long(), 0], eval=True, eval_seed=eval_seed).to(device)
            src, pos_dst, neg_dst, edge_index, msg, x = neighbor_loader(batch['edge_index'].T, neg_dst, 'train')
            pos_out, neg_out = model(batch=batch, src=src.to(device), pos_dst=pos_dst.to(device), neg_dst=neg_dst.to(device),
                                     msg=msg.to(device), x=x.to(device), edge_index=edge_index.to(device))
        if neg_sampler is None:
            y_true = batch.y.cpu()
            y_pred = pos_out.detach().cpu()
            y_pred_list.append(y_pred)
        else:
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
            y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))],
                               dim=0)    
            y_pred_list.append((y_pred.sigmoid() > 0.5).float())

        y_pred_confidence_list.append(y_pred)
        y_true_list.append(y_true)
        if hasattr(batch, "hash_id") or (isinstance(batch, dict) and "hash_id" in batch):
            hash_id_list.append(batch['hash_id'])
            #positive samples
            malicious_list.append(batch['malicious'])
            #negative samples
            malicious_list.append(batch['malicious'])

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, static_x)
        neighbor_loader.insert(src.cpu(), pos_dst.cpu())

    t1 = time.time()

    # Compute scores  
    y_true_list = torch.cat(y_true_list).unsqueeze(1)
    y_pred_list = torch.cat(y_pred_list)
    y_pred_confidence_list = torch.cat(y_pred_confidence_list)
    if hasattr(batch, "hash_id") or (isinstance(batch, dict) and "hash_id" in batch):
        hash_id_list = torch.cat(hash_id_list)
        malicious_list = torch.cat(malicious_list).bool().cpu()
        scores = scoring(y_true_list[~malicious_list], y_pred_list[~malicious_list], y_pred_confidence_list[~malicious_list], is_regression=neg_sampler is None)
        scores['loss'] = criterion(y_pred_confidence_list[~malicious_list], y_true_list[~malicious_list]).item()
    else:
        scores = scoring(y_true_list, y_pred_list, y_pred_confidence_list, is_regression=neg_sampler is None)
        scores['loss'] = criterion(y_pred_confidence_list, y_true_list).item()  
    scores['time'] = datetime.timedelta(seconds=t1 - t0)

    true_values = (y_true_list, y_pred_list, y_pred_confidence_list.sigmoid(), hash_id_list) if return_predictions else None
    if wandb_log:
        for k, v in scores.items():
            if  k == 'confusion_matrix':
                continue
            else:
                wandb.log({f"{eval_name} {k}, {neg_sampler}":v if k != 'time' else v.total_seconds()}, commit=False)
                
        _cm_name = f"conf_mat {eval_name}, {neg_sampler}"
        _cm = wandb.plot.confusion_matrix(preds=y_pred_list.squeeze().numpy(),
                                          y_true=y_true_list.squeeze().numpy(),
                                          class_names=["negative", "positive"],
                                          title=_cm_name)
        wandb.log({_cm_name : _cm}, commit='val' in eval_name or 'test' in eval_name)
        
    return scores, true_values


@ray.remote(num_cpus=int(os.environ.get('NUM_CPUS_PER_TASK', 1)), num_gpus=float(os.environ.get('NUM_GPUS_PER_TASK', 0.)))
def link_prediction(model_instance, conf):
    return link_prediction_single(model_instance, conf) 


def link_prediction_single(model_instance, conf):
    if conf['wandb']:
        run = wandb.init(project=conf['data_name'], group=str(conf['conf_id']), config=conf, name=conf['save_dir'].split("/")[-1], dir='/mnt/ray-data/kuba/RESULTS' if conf['cluster'] else '.')

    # Set the configuration seed
    set_seed(conf['seed'])
    if not conf['cpu']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    data, data_metadata = get_dataset(root=conf['data_dir'], name=conf['data_name'], version=conf['version'], seed=conf['exp_seed'], metadata= conf['model_params']['hetero_gnn'] if 'hetero_gnn' in conf['model_params'] else False )
    if conf['version'] == 'static':
        train_mess_data, train_sup_data, val_data, test_data = data
        train_data = merge_static_data(train_mess_data, train_sup_data)
        train_val_data = merge_static_data(train_data, val_data)

        if conf['sampler']['disjoint_train']:
            train_loader = LinkStaticLoader(train_sup_data.edge_index, train_sup_data.edge_attr, train_sup_data.hash_id, train_sup_data.malicious, batch_size=conf['batch'])
        else:
            train_mess_data = train_data
            train_loader = LinkStaticLoader(train_data.edge_index, train_data.edge_attr, train_data.hash_id, train_data.malicious, batch_size=conf['batch'])
        val_loader = LinkStaticLoader(val_data.edge_index, val_data.edge_attr, val_data.hash_id, val_data.malicious, batch_size=conf['batch'])
        test_loader = LinkStaticLoader(test_data.edge_index, test_data.edge_attr, test_data.hash_id, test_data.malicious, batch_size=conf['batch'])

        if not conf['sampler']['directed']:
            if conf['sampler']['disjoint_train']:
                train_mess_data = to_undirected(train_mess_data)
            train_data = to_undirected(train_data)
            train_val_data = to_undirected(train_val_data)
            # TODO: make this automatic
            conf['model_params']['num_relations'] +=  27 # =*2
        neighbor_loader = StaticNeighborLoader(train_mess_data, train_data, train_val_data, num_nodes=[conf['sampler']['size']] * conf['model_params']['num_layers'], subgraph_type=True)
    else:
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

    # Define model
    if conf['debug'] and conf['model'] == 'TGN':
        conf['model_params']['log'] = conf['wandb']
    if 'hetero_gnn' in conf['model_params']:
        model = model_instance(**conf['model_params'], data_metadata=data_metadata).to(device)
    else:
        model = model_instance(**conf['model_params']).to(device)
    criterion = torch.nn.BCEWithLogitsLoss() if not conf['link_regression'] else REGRESSION_SCORES[conf['metric']] 
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['optim_params']['lr'], weight_decay=conf['optim_params']['wd'])

    (train_src_nodes, train_dst_nodes, 
     val_src_nodes, val_dst_nodes, 
     test_src_nodes,test_dst_nodes) = get_node_sets(conf['strategy'], train_data, val_data, test_data)

    if conf['link_regression']:
        train_neg_link_sampler = None
        val_neg_link_sampler = None
        test_neg_link_sampler = None
    else:
        neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
        train_neg_link_sampler = neg_sampler_instance(train_src_nodes, train_dst_nodes, data.x[train_dst_nodes, 0], name='train', 
                                                      check_link_existence=not conf['no_check_link_existence'],
                                                      seed=conf['exp_seed']+1)
        val_neg_link_sampler = neg_sampler_instance(val_src_nodes, val_dst_nodes, data.x[val_dst_nodes, 0], name='val', 
                                                    check_link_existence=not conf['no_check_link_existence'],
                                                    seed=conf['exp_seed']+2)
        test_neg_link_sampler = neg_sampler_instance(test_src_nodes, test_dst_nodes, data.x[test_dst_nodes, 0], name='test', 
                                                     check_link_existence=not conf['no_check_link_existence'],
                                                     seed=conf['exp_seed']+3)

    history = []
    best_epoch = 0
    best_score = -np.inf
    data_name = conf['data_name'].split("_")[-1]
    conf_id = conf['conf_id']
    strategy = conf['strategy']
    # Load previuos ckpt if exists
    path_save_best = os.path.join(conf['ckpt_path'], f'conf_{conf["conf_id"]}_seed_{conf["seed"]}.pt')
    epoch_times = []
    if os.path.exists(path_save_best) and not conf['overwrite_ckpt']:
        # Load the existing checkpoint
        print(f'Loading {path_save_best}')
        ckpt = torch.load(path_save_best, map_location=device)
        best_epoch = ckpt['epoch']
        best_score = ckpt['best_score']
        history = ckpt['history']
        epoch_times = ckpt['epoch_times']
        if ckpt['train_ended'] and not conf['inference']:
            # The model was already trained, then return
            return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_to(optimizer, device) # Map the optimizer to the current device    
    model.to(device)
    
    if not conf['inference']:
        for e in range(best_epoch, conf['epochs']):
            t0 = time.time()
            if conf['debug']: print('Epoch {:d}:'.format(e))

            train(data=data, model=model, optimizer=optimizer, train_loader=train_loader, criterion=criterion, 
                neighbor_loader=neighbor_loader, train_neg_sampler=train_neg_link_sampler, helper=assoc, device=device,
                backward=not conf['model']=='EdgeBank', static='static' in conf['version'], conf=conf)
            
            model.reset_memory()
            neighbor_loader.reset_state()

            tr_scores, _ = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                                neighbor_loader=neighbor_loader, neg_sampler=train_neg_link_sampler, helper=assoc, 
                                eval_seed=conf['exp_seed'], device=device, eval_name='train', wandb_log=conf['wandb'], static='static' in conf['version'])
            
            if conf['reset_memory_eval']:
                model.reset_memory()

            vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                            neighbor_loader=neighbor_loader, neg_sampler=val_neg_link_sampler, 
                                            helper=assoc, eval_seed=conf['exp_seed'], device=device,
                                            eval_name='val', wandb_log=conf['wandb'], static='static' in conf['version'])
            history.append({
                'train': tr_scores,
                'val': vl_scores
            })
            epoch_times.append(time.time()-t0)
            if vl_scores[conf['metric']] > best_score:
                best_score = vl_scores[conf['metric']]
                best_epoch = e
                torch.save({
                    'train_ended': False,
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'loss': (tr_scores['loss'], vl_scores['loss'], None),
                    'tr_scores': tr_scores,
                    'vl_scores': vl_scores,
                    'true_values': (None, vl_true_values, None),
                    'history': history,
                    'epoch_times': epoch_times
                }, path_save_best)

            if conf['debug']: print(f'\ttrain :{tr_scores}\n\tval :{vl_scores}')

            if conf['debug'] or (conf['verbose'] and e % conf['patience'] == 0): 
                print(f'Epoch {e}: {np.mean(epoch_times)} +/- {np.std(epoch_times)} seconds per epoch') 

            if e - best_epoch > conf['patience']:
                break

    # Evaluate on test
    if conf['debug']: print('Loading model at epoch {}...'.format(best_epoch))
    ckpt = torch.load(path_save_best, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    ckpt['test_score'] = {}
    ckpt['val_score'] = {}
    ckpt['train_score'] = {}
    ckpt['true_values'] = {}
    ckpt['loss'] = {}

    if conf['use_all_strategies_eval']:
        strategies = dst_strategies
    else:
        strategies = [conf['strategy']]

    for strategy in strategies:
        if conf['link_regression']:
            tmp_train_neg_link_sampler = None
            tmp_val_neg_link_sampler = None
            tmp_test_neg_link_sampler = None
        elif strategies == conf['strategy']:
            tmp_train_neg_link_sampler = train_neg_link_sampler
            tmp_val_neg_link_sampler = val_neg_link_sampler
            tmp_test_neg_link_sampler = test_neg_link_sampler
        else:
            (tmp_train_src_nodes, tmp_train_dst_nodes, 
             tmp_val_src_nodes, tmp_val_dst_nodes, 
             tmp_test_src_nodes, tmp_test_dst_nodes) = get_node_sets(strategy, train_data, val_data, test_data)

            neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
            tmp_train_neg_link_sampler = neg_sampler_instance(tmp_train_src_nodes, tmp_train_dst_nodes, data.x[tmp_train_dst_nodes, 0],
                                                              check_link_existence=not conf['no_check_link_existence'],
                                                              name='train', seed=conf['exp_seed']+1)
            tmp_val_neg_link_sampler = neg_sampler_instance(tmp_val_src_nodes, tmp_val_dst_nodes, data.x[tmp_val_dst_nodes, 0],
                                                            check_link_existence=not conf['no_check_link_existence'],
                                                            name='val', seed=conf['exp_seed']+2)
            tmp_test_neg_link_sampler = neg_sampler_instance(tmp_test_src_nodes, tmp_test_dst_nodes, data.x[tmp_test_dst_nodes, 0],
                                                             check_link_existence=not conf['no_check_link_existence'],
                                                             name='test', seed=conf['exp_seed']+3)

        model.reset_memory()
        neighbor_loader.reset_state()

        tr_scores, tr_true_values = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_train_neg_link_sampler, 
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='train_best', wandb_log=conf['wandb'], return_predictions=conf['return_predictions'], static='static' in conf['version'])
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_val_neg_link_sampler, 
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='val_best', wandb_log=conf['wandb'], return_predictions=conf['return_predictions'], static='static' in conf['version'])
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        ts_scores, ts_true_values = eval(data=data, model=model, loader=test_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_test_neg_link_sampler, 
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='test_best', wandb_log=conf['wandb'], return_predictions=conf['return_predictions'], static='static' in conf['version'])

        ckpt['test_score'][strategy] = ts_scores
        ckpt['val_score'][strategy] = vl_scores
        ckpt['train_score'][strategy] = tr_scores
        ckpt['true_values'][strategy] = (tr_true_values, vl_true_values, ts_true_values)
        ckpt['loss'][strategy] = (tr_scores['loss'], vl_scores['loss'], ts_scores['loss'])
        if 'darpa' in conf['data_name']:
            ts_true_values = ts_true_values[2].squeeze()[ts_true_values[0].bool().squeeze()].numpy(), ts_true_values[3].cpu().squeeze().numpy()
            df = pd.DataFrame({'hash_id':ts_true_values[1], 'prob':ts_true_values[0]})
            data_name = conf['data_name'].split("_")[-1]
            conf_id = conf['conf_id']
            results_path = os.path.join(conf['ckpt_path'], f"{strategy}_conf_{conf_id}_detection_results-{data_name}_seed_{conf['seed']}.csv")
            df.to_csv(results_path)

    ckpt['train_ended'] = True
    torch.save(ckpt, path_save_best)
    if conf['wandb']:
        run.finish()

    return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf
