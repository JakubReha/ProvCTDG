from gqlalchemy import Memgraph
from multiprocessing import Process, Queue
from queue import Empty
from tqdm import tqdm
from pathlib import Path
import sys
import logging
import time
import argparse
from argparse import ArgumentParser
import pandas as pd
import yaml

##############
## CONSTATS ##
##############
ATK_CATS = ["success", "attempt", "all"]


###############
## FUNCTIONS ##
###############
def read_attack_yaml(fpath: str) -> dict:
    with open(fpath, "r") as fin:
        return yaml.safe_load(fin)


def build_dfs_query(
    qtemplate: str, dep: int, edgefile: int, tstart: int, tend: int, hid: str
) -> str:
    equery = qtemplate % (
        dep,
        edgefile,
        tstart,
        tend,
        hid,
        dep,
        edgefile,
        tstart,
        tend,
    )
    return equery


def mp_query(Q: Queue, ip: str, port: int, query: str) -> None:
    try:
        memcon = new_mem_connection(ip, port)
        res = list(memcon.execute_and_fetch(query))[0]
        Q.put(res["edges"])
    except Exception as e:
        logging.error("Exception = %s for query = %s", e, query)
        Q.put(dict())


def new_mem_connection(ip: str, port: int) -> Memgraph:
    return Memgraph(host=ip, port=port)


def get_initial_node_set(
    memgraph: Memgraph, tstart: int, tend: int, atkdict: dict
) -> list[str]:
    # Find initial set of nodes likely related to the attack
    edgefile = atkdict["edge_file"]
    qtemplate = atkdict["nodesq"]
    nquery = qtemplate % (edgefile, tstart, tend)
    logging.debug("nquery = %s", nquery)
    res = list(memgraph.execute_and_fetch(nquery))[0]
    logging.debug("nodeids = %s", res)
    startids = res["uniqhashes"]
    logging.info("Found %d as starting nodes to build attack graph", len(startids))
    del memgraph  # close connection to memgraph db
    return startids


def dfs_search_and_extra(
    args,
    Q: Queue,
    atkdict: dict,
    startids: list[str],
) -> dict:
    qtemplate = atkdict["edgesq"]
    edgefile = atkdict["edge_file"]
    mdep = args.max_dfs_depth
    if "mdep" in atkdict:
        logging.warning("Overwriting command-line paramenter for mdep")
        mdep = atkdict["mdep"]
    logging.info("Using DFS depth = %d", mdep)
    edges = {}
    #########
    ## DFS ##
    #########
    for hid in tqdm(startids):
        currdep = mdep
        while currdep > 0:
            equery = build_dfs_query(qtemplate, currdep, edgefile, tstart, tend, hid)
            p = Process(target=mp_query, args=(Q, args.db_ip, args.db_port, equery))
            p.start()
            try:
                edge_batch = Q.get(timeout=args.timeout)
                edges.update(edge_batch)
                logging.debug("edge_batch = %s", edge_batch)
                p.join()
                p.close()
                break
            except Empty:
                p.kill()
                p.join()
                currdep -= 1
                if currdep > 0:
                    logging.info(
                        "Re-trying DFS search for hid=%s @ depth=%d", hid, currdep
                    )
                else:
                    logging.info("Skipping DFS search for hid=%s", hid)
                time.sleep(0.5)
    ###########################
    ## Process extra queries ##
    ###########################
    if "extraq" in atkdict:
        logging.info("Executing extra memgraph queries")
        for query in tqdm(atkdict["extraq"]):
            p = Process(target=mp_query, args=(Q, args.db_ip, args.db_port, query))
            p.start()
            try:
                edge_batch = Q.get(timeout=args.timeout_extra)
                edges.update(edge_batch)
                logging.debug("edge_batch = %s", edge_batch)
                p.join()
                p.close()
            except Empty:
                p.kill()
                p.join()
                logging.error("Skipping query=%s", query)
                time.sleep(0.5)
    return edges


def _create_pd_from_edge_dict(edges: dict) -> pd.DataFrame:
    df = pd.DataFrame(edges.items(), columns=["edge_hash_id", "srcnode_hash_id"])
    return df


def merge_results(res_dict: dict) -> pd.DataFrame:
    label = "label"
    isall = "all" in res_dict
    issucc = "success" in res_dict
    isattempt = "attempt" in res_dict
    if isall:
        eve = _create_pd_from_edge_dict(res_dict["all"])
    if issucc:
        succ = _create_pd_from_edge_dict(res_dict["success"])
        succ = succ.assign(label="success")
    if isattempt:
        attmpt = _create_pd_from_edge_dict(res_dict["attempt"])
        attmpt = attmpt.assign(label="failure")
    # merge depending on what is available
    if issucc and isall:
        final = pd.merge(eve, succ, on=["edge_hash_id", "srcnode_hash_id"], how="left")
        final = final.fillna("failure")
    elif issucc and isattempt:
        final = pd.concat([attmpt, succ])
    elif issucc:
        final = succ
    else:
        raise ValueError("Unknown combination of categories")
    return final


def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--attack-file",
        "-a",
        required=True,
        type=str,
        help="Yaml file with the attack specification",
    )
    parser.add_argument(
        "--output-root",
        nargs="?",
        default="/root/jakub-reha/ShadeWatcher/data/memgraph/labelling",
        type=str,
        help="Root folder where the csv are stored",
    )
    parser.add_argument(
        "--max-dfs-depth",
        nargs="?",
        default=5,
        type=int,
        help="Max depth for DFS algorithm, can be overwritten by attack-file",
    )
    parser.add_argument(
        "--db-port",
        nargs="?",
        default=7689,
        type=int,
        help="Port to communicate to memgraph DB from Python API",
    )
    parser.add_argument(
        "--db-ip",
        nargs="?",
        default="127.0.0.1",
        type=str,
        help="IP of socket where memgraph DB is listening",
    )
    parser.add_argument(
        "--timeout",
        nargs="?",
        default=20,
        type=int,
        help="Timeout for memgraph DFS query before trying with a lower depth",
    )
    parser.add_argument(
        "--timeout-extra",
        nargs="?",
        default=300,
        type=int,
        help="Timeout for memgraph extra queries specified in attack-file yaml",
    )
    parser.add_argument(
        "--log-file",
        nargs="?",
        default="log_extract_attack.log",
        type=str,
        help="Path of file to be used for logs",
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    return parser


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_args(parser)
    args = parser.parse_args()
    # Logging
    loglvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_file, mode="w"),
        ],
        format="%(asctime)s,%(msecs)d | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=loglvl,
    )
    # Connect to memgraph
    memgraph = Memgraph(host=args.db_ip, port=args.db_port)
    logging.info("Connected to memgraph")

    # Read YAML with query specs for attack
    atkdict = read_attack_yaml(args.attack_file)
    # Set-up some useful variables
    atkname = atkdict["attack_name"]
    tdict = atkdict["time"]
    # simple check
    assert all(
        [key in ATK_CATS for key in tdict.keys()]
    ), "Unkown category in .yaml attack-file for time"

    logging.info("Analyzing %s attack", atkname)
    Q = Queue()
    res_dict = {}
    for attk_cat in tdict:
        tcatdict = tdict[attk_cat]
        logging.info("Extracting edges for category = %s", attk_cat)
        tstart = tcatdict["start"]
        tend = tcatdict["end"]
        startids = get_initial_node_set(memgraph, tstart, tend, atkdict)
        logging.info("Starting backward/forward search from each initial node found")
        atk_cat_edges = dfs_search_and_extra(args, Q, atkdict, startids)
        res_dict[attk_cat] = atk_cat_edges
    # Merge results
    logging.info("Merging the different category results for %s", atkname)
    final = merge_results(res_dict)
    # Write df as csv to disk
    outpath = Path(args.output_root) / f"{atkname}.csv"
    logging.info("Saving results @ %s", outpath)
    final.to_csv(outpath, index=False, header=True)
    logging.info("DONE!")
