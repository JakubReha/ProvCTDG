import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Process
import shutil
import math
import torch
import dask.dataframe as dd
import argparse
import ipaddress
import re
import hashlib
import dask


ports_dict = {
 '5353': 0,
 '53': 1,
 '80': 2,
 '443': 3,
 '123': 4,
 '22': 5,
 '0': 6,
 '67': 7,
 '143': 8,
 '25': 9}

extensions_dict ={
**dict.fromkeys(["dat"], 0),
**dict.fromkeys(["shtml", "html", "css"], 1),
**dict.fromkeys(["eml"], 2),
**dict.fromkeys(["db", "sqlite"], 3),
**dict.fromkeys(["pyc", "sh", "js", "bin," "run", "py"], 4),
**dict.fromkeys(["conf", "json", "txt", "rdf," "pdf", "xml"], 5),
**dict.fromkeys(["gif", "jpg", "png", "jpeg", "svg"], 6),
**dict.fromkeys(["tmp", "bak"], 7),
**dict.fromkeys(["mozlz4", "jsonlz4", "sbstore", "xpt"], 8),
**dict.fromkeys(["so"], 9),
**dict.fromkeys(["gz", "zip", "tar"], 10),
# 11 used to be *others*
**dict.fromkeys(["cc", "c", "cpp", "h"], 11),
}

root_dict = {
 'home': 0,
 'dev': 1,
 'proc': 2,
 'usr': 3,
 'run': 4,
 'var': 5,
 'tmp': 6,
 'etc': 7,
 '(null)': 8}

process_dict = {
 'sudo': 0,
 'sh': 1,
 'bash': 2,
 '-bash': 2,
 'sshd': 3,
 'sshd:': 3,
 'dpkg': 4,
 'apt-config': 5,
 'run-parts': 6,
 'chmod': 7,
 'rm': 8,
 'firefox':9,
 'thunderbird':10,
 'cat':11,
 'whoami':12,
 'python':13,
 'postgres':14,
 'pulseaudio':15,
 'stat': 16,
 'uname':17,
 'console-kit-daemon':18,
 'fluxbox':19
 }

def process_class(process):
    if process in process_dict:
        return process_dict[process]
    else:
        return max(process_dict.values()) + 1

def root_class(root):
    if root in root_dict:
        return root_dict[root]
    else:
        return max(root_dict.values()) + 1

def extension_class(extension_full):
    out = []
    if len(extension_full) == 0:
        # no extension
        return [max(extensions_dict.values()) + 1]
    for key in extensions_dict.keys():
        if key in extension_full.split('.'):
            out.append(extensions_dict[key])
    if len(out) == 0:
        # unknown extension
        return [max(extensions_dict.values()) + 2]
    else:
        return out

def port_class(port):
    if port in ports_dict:
        return ports_dict[port]
    elif int(port) >= 1024 and int(port) <= 49151:
        return max(ports_dict.values()) + 1
    elif int(port) > 49151:
        return max(ports_dict.values()) + 2
    elif int(port) < 1024:
        return max(ports_dict.values()) + 3
    else:
        return max(ports_dict.values()) + 4
    

def check_int(s):
    try:
        int(s)
        return True
    except:
        try:
            int(s[0])
            return True
        except:
            return False
        
def aggregate_edges(nodes, edges):
    print("aggregating edges")
    sockets = nodes.index[nodes.type == 2]
    edges_to_keep = edges[~edges.src.isin(sockets) & ~edges.dst.isin(sockets)]
    edges_to_filter = edges[edges.src.isin(sockets) | edges.dst.isin(sockets)]
    edges_to_filter['src_ip'] = np.nan
    edges_to_filter['dst_ip'] = np.nan
    edges_to_filter.loc[edges_to_filter.src.isin(sockets), 'src_ip'] = nodes.iloc[edges_to_filter.src[edges_to_filter.src.isin(sockets)].values].ip.values
    edges_to_filter.loc[edges_to_filter.dst.isin(sockets), 'dst_ip'] = nodes.iloc[edges_to_filter.dst[edges_to_filter.dst.isin(sockets)].values].ip.values
    edges_to_filter['src'] = edges_to_filter['src'].astype(np.int32)
    edges_to_filter['dst'] = edges_to_filter['dst'].astype(np.int32)
    edges_to_filter['port'] = np.nan
    edges_to_filter.loc[edges_to_filter.src.isin(sockets), 'port'] = nodes.iloc[edges_to_filter.src[edges_to_filter.src.isin(sockets)].values].port.values
    edges_to_filter.loc[edges_to_filter.dst.isin(sockets), 'port'] = nodes.iloc[edges_to_filter.dst[edges_to_filter.dst.isin(sockets)].values].port.values

    edges_to_keep = pd.concat([edges_to_keep, edges_to_filter[edges_to_filter.src_ip.isna() & edges_to_filter.dst_ip.isna()]]).sort_index()
    edges_to_filter = edges_to_filter[~(edges_to_filter.src_ip.isna() & edges_to_filter.dst_ip.isna())]

    
    interval_ms = 1000
    groups = np.zeros((len(edges_to_filter))) - 1
    curr_group = 0
    for column in ['src_ip', 'dst_ip']:
        if column == 'dst_ip':
            node = 'src'
        else:
            node = 'dst'
        df = edges_to_filter[[column, 'time', node, 'ext_roll', 'syscall']].reset_index().dropna()
        index_list = list(df.index)
        time_list = list(df.time)
        ip_list = list(df[column])
        node_list = list(df[node])
        split_list = list(df.ext_roll)
        syscall_list = list(df.syscall)

        for i in tqdm(range(len(df))):
            if groups[index_list[i]] == -1:
                ip_to_match = ip_list[i]
                other_node_to_match = node_list[i]
                start_time = time_list[i]
                split_to_match = split_list[i]
                syscall_to_match = syscall_list[i]
                groups[index_list[i]] = curr_group
                k = i + 1
                if k < len(df):
                    duration = time_list[k] - start_time
                    while duration <= interval_ms:
                        if node_list[k] == other_node_to_match and ip_list[k] == ip_to_match and split_list[k] == split_to_match and syscall_to_match == syscall_list[k]:
                            groups[index_list[k]] = curr_group
                        k += 1
                        if k >= len(df):
                            duration =  interval_ms + 1
                        else:
                            duration = time_list[k] - start_time
                curr_group += 1

    edges_to_filter['groups'] = groups.astype(int)
    grouped_edges_to_be_filtered = edges_to_filter.groupby('groups')
    print("computing unique syscalls per aggregation")
    #unique_syscalls_per_group = list(grouped_edges_to_be_filtered.syscall.unique())
    edges_to_filter = edges_to_filter.drop_duplicates("groups").sort_values('groups')
    #edges_to_filter.syscall = unique_syscalls_per_group
    edges_to_filter = edges_to_filter.sort_index()
    #edges_to_keep.syscall = edges_to_keep.syscall.apply(lambda x: [x])
    edges = pd.concat([edges_to_keep, edges_to_filter]).sort_index()
    print("computing unique ports per aggregation")
    nodes.loc[~nodes.port_class.isna(), 'port_class'] = nodes[~nodes.port_class.isna()].port_class.apply(lambda x: [x])
    for group in tqdm(grouped_edges_to_be_filtered):
        ports = list(set([port_class(port) for port in group[1].port.unique().astype(int)]))
        if isinstance(group[1].dst_ip.iloc[0], float):
            src = group[1].iat[0, 1]
            nodes.at[src, 'port_class'] =  list(set(nodes.at[src, 'port_class'] + ports))
        else:
            dst = group[1].iat[0, 2]
            nodes.at[dst, 'port_class'] = list(set(nodes.at[dst, 'port_class'] + ports))
    return nodes, edges



def parse_data(graph_folder, save_folder, edge_files, test_start, preprocessed, ground_truth_folder):
    #paths = os.listdir(os.path.join(graph_folder, 'edges'))
    paths = [i for i in os.listdir(graph_folder) if "edgefact_tmp" in i]
    dfs = []
    print("reading edges")
    for path in tqdm(paths):
        #dfs.append(dd.read_csv(os.path.join(graph_folder, 'edges', path), header=None))
        dfs.append(pd.read_csv(os.path.join(graph_folder, path), header=None))
        dfs[-1]['file'] = int(path.split('.')[0].split('_')[-1]) 
    edges = pd.concat(dfs, axis=0)
    print("number of edges before deduplication: ", len(edges))
    edges = edges.drop_duplicates([0, 4])
    print("number of unique edges: ", len(edges))
    edges = edges.rename(columns={0: "hash_id", 1: "src", 2:"dst", 3:"syscall", 4:"sequence", 5:"session", 6:"time"})
    unique_nodes = set(edges["src"].values).union(edges["dst"].values)
    print("reading nodes")
    if preprocessed:
        files = pd.read_csv(os.path.join(graph_folder, 'filefact.txt'), index_col=0).drop_duplicates()
        processes = pd.read_csv(os.path.join(graph_folder, 'procfact.txt'), index_col=0).drop_duplicates()
        sockets = pd.read_csv(os.path.join(graph_folder, 'socketfact.txt'), index_col=0).drop_duplicates()
    else:
        colnames=['n_id', 'name', 'version']    
        files = pd.read_csv(os.path.join(graph_folder, 'filefact.txt'), header=None, names=colnames).drop_duplicates()
        #files.to_csv(os.path.join(graph_folder, 'filefact.txt'))
        colnames=['n_id', 'pid', 'exe', 'ppid', 'args']    
        processes = pd.read_csv(os.path.join(graph_folder, 'procfact.txt'), header=None, sep="~~", names=colnames, engine='python', dtype={'pid': 'float64','ppid': 'float64'}).drop_duplicates()
        processes["exe"] = processes.exe.map(lambda x: str(x).split(" ")[0].split("/")[-1])
        #processes.to_csv(os.path.join(graph_folder, 'procfact.txt'))
        colnames=['n_id', 'name']    
        sockets = pd.read_csv(os.path.join(graph_folder, 'socketfact.txt'), header=None, names=colnames).drop_duplicates()
        #sockets.to_csv(os.path.join(graph_folder, 'socketfact.txt'))
    files['type'] = 0
    processes['type'] = 1
    sockets['type'] = 2


    files = files[files.n_id.isin(unique_nodes)]
    processes = processes[processes.n_id.isin(unique_nodes)]
    sockets = sockets[sockets.n_id.isin(unique_nodes)]

    # Textual features
    print("extracting features")
    sockets['ip'] = sockets.name.map(lambda x: ":".join(x.split(":")[:-1]))
    sockets['port'] = sockets.name.map(lambda x: x.split(":")[-1])
    private = []
    for ip in sockets.ip.values:
        try:
            private.append(int(ipaddress.ip_address(str(ip)).is_private))
        except:
            private.append(2)
    sockets['private'] = private
    sockets['port_class'] = sockets.port.map(port_class)
    sockets.loc[(sockets.port=='0') & (sockets.name == ':0'), 'port_class'] = max(ports_dict.values()) + 2
    files['extension'] = files.name.map(lambda x: ".".join(x.split("/")[-1].split(".")[1:]))
    files['extensions_class'] = files.extension.map(extension_class)
    def get_root(x):
        try:
            return x.split('/')[1]
        except:
            return None
    files['root'] = files.name.map(get_root)
    files['root_class'] = files.root.map(root_class)
    processes['processes_class'] = processes.exe.map(process_class)
    nodes = pd.concat([files, processes, sockets]).reset_index(drop=True)

    edges = edges[(edges.file >= edge_files[0]) & (edges.file <= edge_files[1])]
    edges = edges.sort_values("time").reset_index(drop=True)
    edges["time"] = edges["time"] -  edges["time"].min()
    edges.time = edges.time / 1e6
    res = {nodes.n_id.values[i]: nodes.index[i] for i in range(len(nodes.n_id.values))}
    edges['src'] = list(map(lambda x : res[x], list(edges.src.values)))
    edges['dst'] = list(map(lambda x : res[x], list(edges.dst.values)))
    # Train/Test split
    val_time = list(np.quantile(edges[edges["file"] < test_start]["time"].values, [0.85]))[0]
    edges.loc[edges["time"] < val_time, "ext_roll"] = 0
    edges.loc[edges["time"] >= val_time, "ext_roll"] = 1
    edges.loc[edges["file"] >= test_start, "ext_roll"] = 2
    edges["ext_roll"] = edges["ext_roll"].astype(int)
    nodes, edges = aggregate_edges(nodes, edges)


    if "trace" in graph_folder.lower():
        # Ground Truth
        ## part1
        firefox_backdoor = pd.read_csv(os.path.join(ground_truth_folder,"TC3_trace_firefox_backdoor_final.csv"))
        ## part2
        browser_extension = pd.read_csv(os.path.join(ground_truth_folder,"TC3_trace_browser_extension_final.csv"))
        pine_phishing_exe = pd.read_csv(os.path.join(ground_truth_folder,"TC3_trace_pine_phishing_exe_final.csv"))
        trace_thunderbird_phishing_exe = pd.read_csv(os.path.join(ground_truth_folder,"TC3_trace_thunderbird_phishing_exe_final.csv"))
        attacks = pd.concat([firefox_backdoor, browser_extension, pine_phishing_exe, trace_thunderbird_phishing_exe])
    elif "theia" in graph_folder.lower():
        firefox_backdoor = pd.read_csv(os.path.join(ground_truth_folder,"TC3_theia_firefox_backdoor_final.csv"))
        browser_extension = pd.read_csv(os.path.join(ground_truth_folder,"TC3_theia_browser_extension_final.csv"))
        attacks = pd.concat([firefox_backdoor, browser_extension])

    edges["malicious"] = False
    edges.loc[edges.hash_id.isin(attacks.edge_hash_id), "malicious"] = True

    # Save
    #nodes.to_csv(os.path.join(graph_folder, 'nodefact.txt'))
    print("saving")
    nodes.to_csv(os.path.join(save_folder, "attributed_nodes.csv"))
    edges.to_csv(os.path.join(save_folder, "edges.csv"))
    print(len(edges), " relations written")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--dataset", type=str.lower, required=True, choices=["trace", "theia"])
    parser.add_argument("--preprocessed", action = "store_true")
    parser.add_argument("--graph_folder", help="Folder containing parsed data (procfact.txt, filefact.txt, ...)", required=True)
    parser.add_argument("--ground_truth_folder", required=True)

    args = parser.parse_args()

    #E3 TRACE
    if args.dataset == "trace":
        edge_files = [0, 210]
        test_start = 125
    elif args.dataset == "theia":
        edge_files = [0, 24]
        test_start = 12


    dataset = args.dataset

    name = "darpa_%s_%dto%d"%(dataset, edge_files[0], edge_files[1])
    current_save_folder = os.path.join(args.save_folder, name)
    if not os.path.exists(current_save_folder):
        os.makedirs(current_save_folder)
    parse_data(args.graph_folder, current_save_folder, edge_files, test_start, args.preprocessed, args.ground_truth_folder)
    print(name, "done")


