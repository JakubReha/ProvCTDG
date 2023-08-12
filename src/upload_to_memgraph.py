from gqlalchemy import Memgraph
import time as t
from natsort import natsorted
import os
import argparse
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm
import shutil
import glob
import numpy as np
import sys
sys.path.append("../darpa_labelling/tools")
from darpa_parse_utils import nanoseconds_to_datetime


def upload_predictions_to_memgraph(data_path, dataset_local, ground_truth_path, predictions_path, upload_name, dataset, port):
    # Predictions
    predictions = pd.read_csv(predictions_path)

    # Ground Truth
    firefox_backdoor = pd.read_csv(ground_truth_path + f"TC3_{dataset}_firefox_backdoor_final_aggregated.csv")
    browser_extension = pd.read_csv(ground_truth_path + f"TC3_{dataset}_browser_extension_final_aggregated.csv")
    browser_extension = pd.merge(browser_extension, predictions, left_on='edge_hash_id', right_on='hash_id', how='left')
    firefox_backdoor = pd.merge(firefox_backdoor, predictions, left_on='edge_hash_id', right_on='hash_id', how='left')
    if dataset == "trace":
        edges = pd.read_csv(os.path.join(data_path, "darpa_trace_0to210/edges.csv"))
        nodes = pd.read_csv(os.path.join(data_path, "darpa_trace_0to210/attributed_nodes.csv"))
        pine_phishing_exe = pd.read_csv(ground_truth_path + f"TC3_{dataset}_pine_phishing_exe_final_aggregated.csv")
        trace_thunderbird_phishing_exe = pd.read_csv(ground_truth_path + f"TC3_{dataset}_thunderbird_phishing_exe_final_aggregated.csv")
        pine_phishing_exe = pd.merge(pine_phishing_exe, predictions, left_on='edge_hash_id', right_on='hash_id', how='left')
        trace_thunderbird_phishing_exe = pd.merge(trace_thunderbird_phishing_exe, predictions, left_on='edge_hash_id', right_on='hash_id', how='left')
        attacks = pd.concat([firefox_backdoor, browser_extension, pine_phishing_exe, trace_thunderbird_phishing_exe])
        attacks_dict = {f"TC3_{dataset}_firefox_backdoor_final_aggregated":firefox_backdoor, f"TC3_{dataset}_browser_extension_final_aggregated":browser_extension, f"TC3_{dataset}_thunderbird_phishing_exe_final_aggregated":trace_thunderbird_phishing_exe, f"TC3_{dataset}_pine_phishing_exe_final_aggregated":pine_phishing_exe}

    else:
        edges = pd.read_csv(os.path.join(data_path,"darpa_theia_0to25/edges.csv"))
        nodes = pd.read_csv(os.path.join(data_path,"darpa_theia_0to25/attributed_nodes.csv"))
        attacks = pd.concat([firefox_backdoor, browser_extension])
        attacks_dict = {f"TC3_{dataset}_firefox_backdoor_final_aggregated":firefox_backdoor, f"TC3_{dataset}_browser_extension_final_aggregated":browser_extension}

    os.makedirs(os.path.join(dataset_local, "predictions", dataset.upper()), exist_ok=True)
    for key in attacks_dict:
        attack = attacks_dict[key]
        attack.to_csv(os.path.join(dataset_local, "predictions", dataset.upper(), upload_name + "_" + key + "_detections.csv"))

    test_edges_src = edges[edges.ext_roll==2].src
    predictions['srcnode_hash_id'] = nodes.iloc[test_edges_src].n_id.values
    false_positives = predictions[(predictions.prob < 0.5) & ~predictions.hash_id.isin(attacks.edge_hash_id)]
    npartitions = len(false_positives) // 1000
    false_positives = dd.from_pandas(false_positives, npartitions=npartitions)
    false_positives.to_csv(os.path.join(dataset_local, "predictions", dataset.upper(), upload_name + "_" + key + "_false_positives_*.csv"))

    memgraph = Memgraph(host='127.0.0.1', port=port)
    for key in tqdm(attacks_dict):
        query = """LOAD CSV FROM \"/data/predictions/%s/%s_%s_detections.csv\" WITH HEADER IGNORE BAD AS row
                        MATCH (n:Node {hash_id: row.srcnode_hash_id})-[rel:EDGE {attack_aggregated: "%s"}]->() WHERE rel.hash_id = row.edge_hash_id
                        SET rel.%s_pred_probs = row.prob """ %(dataset.upper(), upload_name, key, key, upload_name)
                    
        memgraph.execute(query)

    #false positives
    for i in tqdm(range(npartitions)):
        query = """LOAD CSV FROM \"/data/predictions/%s/%s_%s_false_positives_%s.csv\" WITH HEADER IGNORE BAD AS row
                            MATCH (n:Node {hash_id: row.srcnode_hash_id})-[rel:EDGE]->() WHERE rel.hash_id = row.hash_id
                            SET rel.%s_false_positive = row.prob """ %(dataset.upper(), upload_name, key, f"{i:02}", upload_name)
                        
        memgraph.execute(query)
        
        
        
def upload_ground_truth_to_memgraph(ground_truth_path, dataset_local, dataset, port):
    os.makedirs(os.path.join(dataset_local, "labelling"), exist_ok=True)
    files = [ i for i in os.listdir(ground_truth_path) if os.path.isfile(os.path.join(ground_truth_path, i))]
    for fname in files:
        shutil.copy2(os.path.join(ground_truth_path, fname), os.path.join(dataset_local, "labelling"))

    memgraph = Memgraph(host='127.0.0.1', port=port)
    query = """ MATCH (n:Node)-[rel:EDGE]->()
                SET rel.label = "" """ 
    memgraph.execute(query)
    
    files = [ i for i in os.listdir(ground_truth_path) if i.endswith("aggregated.csv") and dataset in i]
    for file in tqdm(files):
        query = """LOAD CSV FROM \"/data/labelling/%s\" WITH HEADER IGNORE BAD AS row
                    MATCH (n:Node {hash_id: row.srcnode_hash_id})-[rel:EDGE]->() WHERE rel.hash_id = row.edge_hash_id
                    SET rel.label = row.label
                    SET rel.attack_aggregated = \"%s\"""" %(file, file.split(".")[0].split('-')[0])   
        memgraph.execute(query)
    

def upload_dataset_to_memgraph(dataset_local, dataset_memgraph, port):
    """
    Delete data from the database and populate from scratch
    """
    st = t.time()
    memgraph = Memgraph(host='127.0.0.1', port=port)
    print("Deleting the contents of the database")
    # Delete all nodes and relationships
    query = "MATCH (n) DETACH DELETE n"
    # Execute the query
    memgraph.execute(query)
    
    query = "FREE MEMORY"
    memgraph.execute(query)
    print("Uploading dataset to Memgraph")
    # Create a node with the label FirstNode and message property with the value "Hello, World!"
    query = """CREATE INDEX ON:Node(hash_id)"""
    memgraph.execute(query)
    query = """CREATE INDEX ON:Node(type)"""
    memgraph.execute(query)
    #query = """CREATE CONSTRAINT ON (n:Node) ASSERT n.hash_id IS UNIQUE"""
    #memgraph.execute(query)
    query = """LOAD CSV FROM \"%s\" NO HEADER IGNORE BAD AS row
    CREATE (n:Node {hash_id:ToString(row[0]), name: row[1], type:\"File\"})"""%(os.path.join(dataset_memgraph, "split/filefact.txt"))
    memgraph.execute(query)
    query = """LOAD CSV FROM \"%s\" NO HEADER IGNORE BAD AS row
    CREATE (n:Node {hash_id:ToString(row[0]), name: row[1], type:\"Socket\"})"""%(os.path.join(dataset_memgraph, "split/socketfact.txt"))
    memgraph.execute(query)
    query = """LOAD CSV FROM \"%s\" NO HEADER IGNORE BAD AS row
    CREATE (n:Node {hash_id:ToString(row[0]), pid: ToInteger(row[1]), name: row[2], ppid: ToInteger(row[3]), type:"Process"})"""%(os.path.join(dataset_memgraph, "split/procfact.txt"))
    memgraph.execute(query)
    query = """CREATE INDEX ON:EDGE(hash_id)"""
    memgraph.execute(query)
    query = """CREATE INDEX ON:EDGE(edge_file)"""
    memgraph.execute(query)
    edge_files = [ i for i in natsorted(os.listdir(dataset_local)) if "edge" in i and 'csv' in i]
    for edge_file in tqdm(edge_files):
        query = """LOAD CSV FROM \"%s\" NO HEADER IGNORE BAD AS row
        MATCH (p1:Node {hash_id: ToString(row[1])})
        MATCH (p2:Node {hash_id: ToString(row[2])})
        CREATE (p1)-[rel:EDGE]->(p2)
        SET rel.hash_id = ToString(row[0])
        SET rel.name = ToInteger(row[3])
        SET rel.sequence = ToInteger(row[4])
        SET rel.timestamp = ToFloat(row[6])
        SET rel.datetime = ToString(row[8])
        SET rel.session = ToInteger(row[5])
        SET rel.edge_file = ToInteger(row[7])""" %(os.path.join(dataset_memgraph, "split", edge_file))
        memgraph.execute(query)
    
    print("Deleting zero degree nodes")
    query = """MATCH (n:Node) WHERE degree(n) = 0 DELETE n"""
    memgraph.execute(query)
    print("Creating a snapshot of the database")
    query = """CREATE SNAPSHOT"""
    memgraph.execute(query)
    end = t.time()
    print("Uploading dataset finished")
    print("Uploading dataset took %ds"%(end-st))
    return memgraph


def split_and_clean_csvs(source_path, redo):
    files = [ i for i in os.listdir(source_path) if 'edge' in i]
    destination_path = os.path.join(source_path, "split")
    if not os.path.exists(destination_path) or redo:
        os.makedirs(destination_path, exist_ok=True)
        old = glob.glob(destination_path + '/*')
        for f in old:
            os.remove(f)
        print("Cleaning and splitting edge files into smaller chunks")
        ddfs = []
        for file in tqdm(files):
            df = pd.read_csv(os.path.join(source_path, file), header=None)
            df["7"] = np.ones(len(df)) * int(file.split(".")[0].split("_")[-1])
            df = dd.from_pandas(df, npartitions=1)
            ddfs.append(df)
        ddf = dd.concat(ddfs)
        ddf = ddf.drop_duplicates(subset=[0, 4])
        size = len(ddf)
        ddf = ddf.repartition(npartitions=int(size/10000))
        print("converting time")
        ddf["8"] = ddf[6].map(nanoseconds_to_datetime)
        print("saving")
        ddf.to_csv(os.path.join(destination_path, "edgefact_small_*.csv"), index=None, header=None)
        print("Splitting edge files finished")
        print(f"Final number of edges: {size}")
        
        files = [ i for i in os.listdir(source_path) if 'fact.txt' in i]
        for file in tqdm(files):
            if 'proc' in file:
                sep = "~~"
                df = pd.read_csv(os.path.join(source_path, file), header=None, sep=sep, usecols=range(0,4)).drop_duplicates()
                df[2] = df[2].map(lambda x: str(x).split(" ")[0].split("/")[-1])
            else:
                df = pd.read_csv(os.path.join(source_path, file), header=None).drop_duplicates()
            df.to_csv(os.path.join(destination_path, file), index=None, header=None)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", type=str.lower, choices=["dataset", "ground_truth", "predictions"])
    parser.add_argument("--dataset_local", help="Dataset folder inside the Memgraph mounted folder in the local filesystem", required=False)
    parser.add_argument("--dataset_memgraph", help="Folder inside the docker container, (/data/e3_trace or /data/e3_theia)", required=True)
    parser.add_argument("--port", default=7687, type=int)
    parser.add_argument("--redo", action="store_true")
    
    parser.add_argument("--dataset", type=str.lower, default="theia", choices=["theia", "trace"])
    # Uploading Ground Truth
    parser.add_argument("--ground_truth_path", required=False)
    # Uploading Predictions
    parser.add_argument("--data_path", help="Path to the folder containing the processed datasets (e.g. data_path/darpa_trace_0to210/edges.csv)" , required=False)
    parser.add_argument("--predictions_path", required=False)
    parser.add_argument("--upload_name", default="TGN_seed_0")
    
    
    args = parser.parse_args()
    if args.type == "dataset":
        split_and_clean_csvs(args.dataset_local, args.redo)
        args.dataset_local = os.path.join(args.dataset_local, "split")
        upload_dataset_to_memgraph(args.dataset_local, args.dataset_memgraph, args.port)
    elif args.type == "ground_truth":
        upload_ground_truth_to_memgraph(args.ground_truth_path, args.dataset_local, args.dataset, args.port)
    elif args.type == "predictions":
        upload_predictions_to_memgraph(args.data_path, args.dataset_local, args.ground_truth_path, args.predictions_path, args.upload_name, args.dataset, args.port)