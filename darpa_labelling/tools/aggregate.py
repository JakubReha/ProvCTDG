import pandas as pd
import os
import argparse

def aggregate(ground_truth_folder, edges_path, dataset):
    edges = pd.read_csv(edges_path)
    if dataset == "trace" or dataset == "theia":
        firefox_backdoor = pd.read_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_firefox_backdoor_final.csv"))
        browser_extension = pd.read_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_browser_extension_final.csv"))
        if dataset == "trace":
            pine_phishing_exe = pd.read_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_pine_phishing_exe_final.csv"))
            trace_thunderbird_phishing_exe = pd.read_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_thunderbird_phishing_exe_final.csv"))
            pine_phishing_exe = pine_phishing_exe[pine_phishing_exe.edge_hash_id.isin(edges.hash_id)]
            trace_thunderbird_phishing_exe = trace_thunderbird_phishing_exe[trace_thunderbird_phishing_exe.edge_hash_id.isin(edges.hash_id)]
            pine_phishing_exe.to_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_pine_phishing_exe_final_aggregated.csv"))
            trace_thunderbird_phishing_exe.to_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_thunderbird_phishing_exe_final_aggregated.csv"))
        
        firefox_backdoor = firefox_backdoor[firefox_backdoor.edge_hash_id.isin(edges.hash_id)]
        browser_extension = browser_extension[browser_extension.edge_hash_id.isin(edges.hash_id)]
        firefox_backdoor.to_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_firefox_backdoor_final_aggregated.csv"))
        browser_extension.to_csv(os.path.join(ground_truth_folder, f"TC3_{dataset}_browser_extension_final_aggregated.csv"))



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_folder", required=True)
    parser.add_argument("--edges_path", required=True)
    parser.add_argument("--dataset", type = str.lower, choices=['trace', 'theia'])
    args = parser.parse_args()