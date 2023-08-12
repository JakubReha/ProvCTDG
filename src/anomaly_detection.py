import numpy as np
import pandas as pd
import argparse
import wandb
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib.pyplot import text
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def compute_detection_performance(prediction_folder, ground_truth_path, model_name, conf_id, dataset, num_seeds, log_wandb, save_folder):
    save_folder = os.path.join(save_folder, model_name)
    os.makedirs(save_folder, exist_ok=True)
    if log_wandb:
        run = wandb.init(project=f"darpa_{dataset}_anomaly_detection", name=model_name)
    if dataset == 'trace':
        split = "0to210"
        # Ground Truth
        ## part1
        firefox_backdoor = pd.read_csv(os.path.join(ground_truth_path,"TC3_trace_firefox_backdoor_final.csv"))
        ## part2
        browser_extension = pd.read_csv(os.path.join(ground_truth_path,"TC3_trace_browser_extension_final.csv"))
        pine_phishing_exe = pd.read_csv(os.path.join(ground_truth_path,"TC3_trace_pine_phishing_exe_final.csv"))
        trace_thunderbird_phishing_exe = pd.read_csv(os.path.join(ground_truth_path,"TC3_trace_thunderbird_phishing_exe_final.csv"))
        attacks_dict = {"firefox_backdoor":firefox_backdoor, "browser_extension":browser_extension, "pine_phishing_exe":pine_phishing_exe, "trace_thunderbird_phishing_exe":trace_thunderbird_phishing_exe}
    elif dataset == 'theia':
        split = "0to25"
        firefox_backdoor = pd.read_csv(os.path.join(ground_truth_path,"TC3_theia_firefox_backdoor_final.csv"))
        browser_extension = pd.read_csv(os.path.join(ground_truth_path,"TC3_theia_browser_extension_final.csv"))
        attacks_dict = {"firefox_backdoor":firefox_backdoor, "browser_extension":browser_extension}
    else:
        raise NotImplemented
    modes = ['success', 'failure']
    keys = list(attacks_dict.keys())
    
    tpr = []
    fpr = []
    auc = []
    ap = []
    attack_detections = {}
    prediction_path = os.path.join(prediction_folder, f"split_conf_{conf_id}_detection_results-{split}_seed_0.csv")
    predictions = pd.read_csv(prediction_path)
    predictions_hashes = set(predictions.hash_id)
    for k in attacks_dict:
        attacks_dict[k] = attacks_dict[k][attacks_dict[k].edge_hash_id.isin(predictions_hashes)]
        attack_detections[k] = {}

    preds_total = np.zeros(len(predictions)) 
    for seed in range(0, num_seeds):
        prediction_path = os.path.join(prediction_folder, f"split_conf_{conf_id}_detection_results-{split}_seed_{seed}.csv")
        predictions = pd.read_csv(prediction_path)
        predictions['attack'] = ['benign'] * len(predictions)
        predictions['mode'] = ['other'] * len(predictions)
        for k in keys:
            mal_mask = predictions["hash_id"].isin(attacks_dict[k]['edge_hash_id'])
            predictions.loc[mal_mask, 'attack'] = k
            
            for mode in modes:
                mask = predictions["hash_id"].isin(attacks_dict[k]['edge_hash_id'][attacks_dict[k]['label'] == mode])
                predictions.loc[mask, 'mode'] = mode
                preds = (1 - predictions.loc[mask].prob.values).round().astype(int)
                y_true = mask.values.astype(int)
                size = len(attacks_dict[k]['edge_hash_id'][attacks_dict[k]['label'] == mode])
                if size > 0:
                    if mode in attack_detections[k]:
                        attack_detections[k][mode].append(preds.sum())
                    else:
                        attack_detections[k][mode] = [preds.sum()]
                    if log_wandb:
                        wandb.log({f"{k} total {size}, {mode}": preds.sum()}, step = 0)

        preds = (1 - predictions.prob.values)
        y_true = (predictions['attack']!='benign').values.astype(int)
        preds_label = preds.round().astype(int)
        fp = (preds_label.astype(bool) & ~y_true.astype(bool)).sum()
        tn = (~preds_label.astype(bool) & ~y_true.astype(bool)).sum()
        tp = (preds_label.astype(bool) & y_true.astype(bool)).sum()
        fn = (~preds_label.astype(bool) & y_true.astype(bool)).sum()
        auc.append(roc_auc_score(y_true, preds))
        ap.append(average_precision_score(y_true, preds))
        fpr.append(fp / (fp + tn))
        tpr.append(tp / ( tp + fn ))
        preds_total += preds / num_seeds


    metrics = {'fpr': fpr, 'auc': auc, 'ap': ap, 'tpr': tpr}
    results = {}
    for metric_name, metric in metrics.items():
        mean = np.array(metric).mean()
        std = np.array(metric).std()
        results[f"{metric_name} total mean"] = [mean]
        results[f"{metric_name} total std"] = [std]
        if log_wandb:
            wandb.log({f"{metric_name} total mean": mean}, step = 0)
            wandb.log({f"{metric_name} total std": std}, step = 0)
    df = DataFrame(results)
    df.to_csv(os.path.join(save_folder, "detection_stats_%s.csv"%model_name))

    if log_wandb:
        _cm_name = f"conf_mat total "
        _cm = wandb.plot.confusion_matrix(preds=preds_label,
                                            y_true=y_true.astype(int),
                                            class_names=["benign", "anomaly"],
                                            title=_cm_name)
        wandb.log({_cm_name : _cm}, step = 0)
        wandb.log({f"roc_curve total": wandb.plot.roc_curve(y_true, np.concatenate((1 - preds_total[:, None], preds_total[:, None]), axis=1), labels=['benign', 'anomaly'])}, step = 0)
        wandb.log({f"pr_curve total": wandb.plot.pr_curve(y_true, np.concatenate((1 - preds_total[:, None], preds_total[:, None]), axis=1), labels=['benign', 'anomaly'])}, step = 0)
        

    malicious = preds_total[y_true==1]
    benign = preds_total[y_true==0]
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(6.4,4.8*1.4))
    plt.rcParams.update({'font.size': 19})
    plt.rcParams['axes.axisbelow'] = True
    plt.hist([malicious, benign], bins, label=['Malicious', 'Benign'], weights=[np.ones(len(malicious))/len(malicious), np.ones(len(benign))/len(benign)], color=['#fe6100', '#648fff'])
    plt.vlines(x=0.5, ymin=0, ymax=1, colors='black', ls='--', lw=2, label='Threshold')
    plt.grid(axis='y')
    plt.ylim(0, 1)
    step = 0.25
    plt.xticks(np.arange(0, 1 + step, step))
    text(0.5, 0.5, " $FPR_{AD}$ : %.1f%% \n $TPR_{AD}$ : %.1f%%"%(np.array(fpr).mean()*100, np.array(tpr).mean()*100), rotation=0, verticalalignment='center')
    plt.legend(loc='upper right')
    plt.xlabel("Anomaly score")
    plt.ylabel("Normalized counts")
    plt.savefig(os.path.join(save_folder, "hist_anom_%s.pdf"%model_name), bbox_inches='tight')

    if log_wandb:
        wandb.log({"threshold": wandb.Image(plt)})
        run.finish()
    print(model_name, "done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prediction_folder', requierd=True)
    parser.add_argument('--ground_truth_path', requierd=True)

    parser.add_argument('--model_name', default="TGN")
    parser.add_argument('--conf_id', default=0)
    parser.add_argument('--dataset', type=str.lower, default='trace', choices=['trace', 'theia'])
    parser.add_argument('--num_seeds', default=5)
    parser.add_argument('--log_wandb', action="store_true")
    parser.add_argument('--save_folder', default="figures/")


    args = parser.parse_args()
    print(args)

    compute_detection_performance(**vars(args))