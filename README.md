# Anomaly Detection in Continuous-Time Temporal Provenance Graphs

Official code repository for the paper "_Anomaly Detection in Continuous-Time Temporal Provenance Graphs_", which was accepted to Temporal Graph Learning Workshop @ NeurIPS 2023.

Please consider citing us

	@inproceedings{
	reha2023anomaly,
	title={Anomaly Detection in Continuous-Time Temporal Provenance Graphs},
	author={Jakub Reha and Giulio Lovisotto and Michele Russo and Alessio Gravina and Claas Grohnfeldt},
	booktitle={Temporal Graph Learning Workshop @ NeurIPS 2023},
	year={2023},
	url={https://openreview.net/forum?id=88tGIxxhsf}
	}
 
## Requirements
_Note: we assume [Miniconda/Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) and [Docker](https://docs.docker.com/engine/install/) are installed. The proper Python version is installed during the first step of the following procedure._

Install the required packages and create the environment with [create_env](./src/create_env.sh) script, the env is called `ctdg_pyg` in it:
``` bash
./create_env.sh 
```

or create the environment from the yml file
``` bash
conda env create -f conda_export_env.yml
conda activate ctdg_pyg 
```

other dependencies: dask, gqalchemy, docker, natsorted, tqdm, wandb

## Table of Contents
We provide pre-processed labeled data with feature extracted [here](https://github.com/JakubReha/ProvCTDG/releases/tag/v1.0.0.0) and the extracted [groundtruth](darpa_labelling/groundtruth/). If you use this data, you can skip steps 1. to 6. and start directly at step 7..

1. Download data (Optional)
2. Data preprocessing (Optional)
4. Visualize data (Optional)
3. Data labelling (Optional)
5. Feature extraction (Optional)
6. Aggregate ground truth (Optional)
7. Visualize the ground truth (Optional)
8. Train and Test (model selection)
9. Anomaly detection

## 1. Download data
### DARPA TC ENGAGEMENT 3
[google drive](https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK)
 
#### Option 1
1) Download [gdrive](https://github.com/prasmussen/gdrive) and setup the service account
2) Run
```
while true; do gdrive download <folder_id> --recursive --skip --service-account <credentials_path>.json && break; done
```
#### Option 2
- can only download 50 files at a time, Google may further limit the amount of downloads
```
gdown https://drive.google.com/drive/folders/<folder_id> --folder --remaining-ok --no-check-certificate
```

## 2. Data preprocessing
See [darpa_preprocessing](darpa_preprocessing)


## 3. Visualize data
Install Memgraph python client.
```
pip install gqlalchemy
```

If you encounter problems during installation you might need to do:
```
# problem related to  OpenSSL
export C_INCLUDE_PATH=<environment_path>/include/
```

or reinstall cmake [link](https://memgraph.github.io/pymgclient/introduction.html#installation)
and then try again.

Then you can start one Memgraph container per dataset: (our code was tested with the version memgraph/memgraph-platform:2.6.5-memgraph2.5.2-lab2.4.0-mage1.6)

```bash
docker run -itd --name trace -p 7688:7687 -p 7445:7444 -p 3001:3000 -v <data_path>:/data memgraph/memgraph-platform:2.6.5-memgraph2.5.2-lab2.4.0-mage1.6
```

```bash
docker run -itd --name theia -p 7689:7687 -p 7446:7444 -p 3002:3000 -v <data_path>:/data memgraph/memgraph-platform:2.6.5-memgraph2.5.2-lab2.4.0-mage1.6
```

Assumed file structure:

```
<data_path>
│
└─── theia
│       procfact.txt 
│       socketfact.txt 
│       filefact.txt 
│       edgefact_tmp_x.txt 
│   
└─── trace
        ...

```

```bash
python upload_to_memgraph.py --dataset_local data_path/dataset_name --dataset_memgraph memgraph_data_path/dataset_name --port 7688
```
When loading edges into the Memgraph database, we split the .csv files containing the edges into smaller chunks. This is done because loading smaller files is much faster than loading big files (this is probably a bug in Memgraph and might be fixed in the future).

Uploading TRACE or THEIA takes ~10 min.

The steps of the script:

1. split edge files into smaller chunks and save inside the 'split' subdirectory
2. delete all contents of the Memgraph databse
3. upload nodes
4. upload edges
5. delete nodes with 0-degree
6. create a snapshot of the database

In your browser go to:
```
localhost:3001
```

- Example of a Memgraph query, Memgraph uses the Cypher language (same as Neo4j)
```
MATCH (n)-[r]->(m) RETURN * LIMIT 1000
```

To change the query timeout, modify the *--query-execution-timeout-sec* parameter in the [memgraph.conf](memgraph.conf) file, put it in the docker container and restart the container:
```
docker cp memgraph.conf trace:/etc/memgraph/memgraph.conf
docker restart trace
```

To use our Graph Style copy the content of [memgraph_graph_style.json](memgraph_graph_style.json) into the Graph Style Editor in your browser.


## 4. Data labelling 
See Memgraph installation and data upload above (**3. Visualize data**).

Then follow [darpa_labelling](darpa_labelling).


## 5. Feature extraction
This extracts features for nodes, aggregates edges (see detailed explanation in our paper) and computes new unique hashes for the edges. 

```bash
python process_data.py --dataset TRACE --save_folder save_folder --ground_truth_folder ground_truth_folder --graph_folder graph_folder
```

## 6. Aggregate ground truth 
The ground truth contains labels for the raw edges. If you want to visualize the ground truth or compute the statistics, aggregation is necessary:

```bash
cd darpa_labelling/tools
python aggregate.py --ground_truth_folder ground_truth_folder --edges_path edges_folder/edges.csv --dataset TRACE
```

## 7. Visualize the ground truth

If you did not execute steps 1. to 6., extract the datasets we provide inside the `DATA` folder (e.g., `/somepath/data/darpa_datasets`)

If you want to upload the ground truth or prediction of a model use:

```
cd src
python upload_to_memgraph.py --ground_truth --port 7688 --dataset TRACE --ground_truth_path ../groundtruth/ --local_path_to_memgraph /memgraph/
```

Note that uploading is quite slow, therefore the code handles uploading only the aggregated ground truth and predictions (step **5. Feature extraction** is necessary).

## 8. Train and Test Link Prediction (model selection)


The datasets in the `DATA` folder are split (ext_roll column in edges.csv) into train(0), validation(1) and test(2) sets in a way that the test set contains all the malicious attacks. The rest is split temporally in the ratio 0.85/0.15 into train and validation sets.

Before training, set the hyperparameters and other settings (OHD-TGN, DIR-TGN, Hetero-TGN, HGT-TGN) in *conf.py*.

Hetero-TGN: 'hetero_gnn' = [True], 'hetero_transformer' = [False]  
HGT-TGN: 'hetero_gnn' = [True], 'hetero_transformer' = [True]  
OHD-TGN: 'one_hot_dir' = [True]  
DIR-TGN: 'dir_GNN' = [True]  

For other hyperameters, the code will run model selection if multiple hyperparameter values are provided. Each combination of hyperparameter values is called a 'configuration' and has a unique 'conf_id' assigned to itself. 

The code can train multiple configurations in parallel using [Ray](https://www.ray.io/) (`--parallelism`). After the validation loss converges (--patience), inference on the test set is performed and the resulting prediction scores are saved into ``` .csv``` files in ```<save_dir>/<model>/ckpt/``` for each configuration and random seed(--num_runs). Note that these prediction scores corresponds to the predicted probability of edge existence. Model selection results (containing all information about individual configurations and scores for all metrics on train, validation and test sets) are saved in the ```<save_dir>/<model>/model_selection_results.csv``` path. When evaluating the test set, the malicious edges are masked out. Model checkpoints are saved in ```<save_dir>/<model>/ckpt/```.

We use [Weights&Biases](https://wandb.ai/site) for logging.

Note that if a dataset is run for the first time, cache files are created in ```<dataset_path>/temporal_processed/``` and ```<dataset_path>/delta_t_stats.pkl/```. Therefore, if something within a dataset is changed later on, please remove the cached files, so they can be computed again on the updated dataset.

**A.** Train CTDHG

```bash
python -u main.py --data_name darpa_theia_0to24 --model TGN --version temporal --parallelism 5 --epochs 50 --batch 200 --save_dir experiment_name --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_experiment_name 2> err_experiment_name
```

**B.** Train Graph Baseline

```bash
python -u main.py --data_name darpa_theia_0to24 --model RGCN --version static --parallelism 5 --epochs 50 --batch 200 --save_dir experiment_name --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_experiment_name 2> err_experiment_name
```

**C.** Train MLP

```bash
python -u main.py --data_name darpa_theia_0to24 --model MLP --version temporal --parallelism 5 --epochs 50 --batch 200 --save_dir experiment_name --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_experiment_name 2> err_experiment_name
```

## 9. Anomaly detection
After you have trained a model you can compute the anomaly scores with the following script:

```bash
python -u anomaly_detection.py --prediction_folder prediction_folder --ground_truth_path path_to_ground_truth --save_folder save_folder --model_name TGN --dataset THEIA --conf_id 0 --wandb
```

