# Darpa labelling

## Groundtruth extraction

After, having completed the [set-up](../README.md#3-visualize-data) of Memgraph, you can use the [extract_attack.py](tools/extract_attack.py) script to extract the attacks groundtruth. That is, label the edges that are part of an attack present in the dataset.

```
usage: extract_attack.py [-h] --attack-file ATTACK_FILE [--output-root [OUTPUT_ROOT]] [--max-dfs-depth [MAX_DFS_DEPTH]] [--db-port [DB_PORT]] [--db-ip [DB_IP]] [--timeout [TIMEOUT]]
                         [--timeout-extra [TIMEOUT_EXTRA]] [--log-file [LOG_FILE]] [--debug | --no-debug]

optional arguments:
  -h, --help            show this help message and exit
  --attack-file ATTACK_FILE, -a ATTACK_FILE
                        Yaml file with the attack specification (default: None)
  --output-root [OUTPUT_ROOT]
                        Root folder where the csv are stored (default: /data/memgraph/labelling)
  --max-dfs-depth [MAX_DFS_DEPTH]
                        Max depth for DFS algorithm, can be overwritten by attack-file (default: 5)
  --db-port [DB_PORT]   Port to communicate to memgraph DB from Python API (default: 7689)
  --db-ip [DB_IP]       IP of socket where memgraph DB is listening (default: 127.0.0.1)
  --timeout [TIMEOUT]   Timeout for memgraph DFS query before trying with a lower depth (default: 20)
  --timeout-extra [TIMEOUT_EXTRA]
                        Timeout for memgraph extra queries specified in attack-file yaml (default: 300)
  --log-file [LOG_FILE]
                        Path of file to be used for logs (default: log_extract_attack.log)
  --debug, --no-debug
```

Example:

```
python extract_attack.py -a attacks_yaml/TC3/trace/browser_extension.yaml --output-root /data/groundtruth/ --db-port 7688 
```

The script takes as input an `--attack-file` (mandatory) and returns a CSV file. This contains: (1) the unique identifier of each edge that is part of the specific attack, (2) the source node id and (3) the label $in$ {success, failure}, where failure indicates that an edge is related to a failed attack attempt. Attack yaml files are structured as follows:

```yaml
attack_name: example_attack
time:  # possible combinations are: (success), (attempt, success), and (all, success)
  attempt:  # start/end timestamp of failed attack attempt
    start: 1523551440000000000
    end: 1523553360000000000
  success:  # successfull attack
    start: 1523553360000000000
    end: 1523553960000000000
edge_file: 20  # edge file where the attack is contained
mdep: 3  # DFS depth for edgesq
# query to find the initial set of nodes realted to an attack from which carrying out DFS-based backward & forward search
nodesq: 'MATCH (n)-[r]->(m) WHERE r.edge_file=%d AND r.timestamp > %d AND r.timestamp RETURN collect(DISTINCT n.hash_id) as uniqhashes'
# backward & forward DFS search carried out starting from each node returned in nodesq
edgesq: "MATCH path=(l)-[* ..%d ( e, l | e.edge_file=%d and e.timestamp > %d AND e.timestamp < %d)]->(c {hash_id: '%s'})-[* ..%d ( p, c | p.edge_file=%d and p.timestamp > %d and p.timestamp < %d)]->(z) UNWIND relationships(path) as allrel WITH DISTINCT allrel as rel RETURN collect(rel.hash_id, startNode(rel).hash_id) as edges"
extraq:  # additional "hard-coded" queries (optional) to find additional edges related to an attack
  - 'MATCH(n)-[r]->(m) WHERE r.timestamp>=1523643600000000000 AND r.timestamp < 1523644200000000000 WITH DISTINCT r as rel RETURN collect(rel.hash_id, startNode(rel).hash_id) as edges'
```

We created YAML attack files for the attacks in TRACE and THEIA ([TC3](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md)) in an iterative approach based on the informations available in the [groundtruth PDF](https://drive.google.com/file/d/1mrs4LWkGk-3zA7t7v8zrhm0yEDHe57QU/view?usp=drive_link).
We make our groundtruth available [here](./groundtruth/), feel free to reach out to us for feedbacks and/or suggestions. 



## Utils for Darpa data analysis

The [darpa_parse_utils.py script](./tools/darpa_parse_utils.py) can be used for supporting the analysis of the both TRACE and THEIA datasets.
For example, it supports in transforming the dates/times shared in the "Event Log" of each attack in the [groundtruth PDF](https://drive.google.com/file/d/1mrs4LWkGk-3zA7t7v8zrhm0yEDHe57QU/view?usp=drive_link) to UTC timestamps present in the data (and viceversa).

```
usage: darpa_parse_utils.py [-h] [--timestamp [TIMESTAMP]] [--date [DATE]] [--edgeid [EDGEID]]

optional arguments:
  -h, --help            show this help message and exit
  --timestamp [TIMESTAMP], -t [TIMESTAMP]
                        Nanosecond timestamp
  --date [DATE], -d [DATE]
                        Format e.g., 2022-03-04_09:46, DARPA timezone
  --edgeid [EDGEID], -e [EDGEID]
                        System call integer
```


## Aggregate
TODO: Kuba


## Upload to memgraph
TODO: Kuba update this

```bash
python upload_to_memgraph.py --ground_truth --port 7688 --dataset TRACE --ground_truth_path ../groundtruth/ --local_path_to_memgraph /mnt/vdb/trace/parsed/memgraph/
```


```bash
python upload_to_memgraph.py --prediction --port 7688 --dataset TRACE --data_path /mnt/vdc/DATA --predictions_path /mnt/vdc/RESULTS/TGN_debug/TGN/ckpt/split_conf_0_detection_results-0to210_seed_0.csv  --ground_truth_path ../groundtruth/ --local_path_to_memgraph /mnt/vdb/trace/parsed/memgraph/ --upload_name TGN_no_memory_seed_0
```