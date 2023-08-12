#!/bin/bash

# Static Baseline
export NUM_GPUS_PER_TASK=0.15
# GAT 
python -u main.py --data_name darpa_0to125 --parallelism 8 --epochs 50 --batch 200 --save_dir GAT_production_0to125_undirected --data_dir DATA --num_runs 1 --patience 5 --overwrite_ckpt --wandb --no_check_link_existence --model GAT --version static > out_gat 2> err_gat #& echo $! > run4.pid
python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir GAT_production_126to210_undirected --data_dir DATA --num_runs 1 --patience 5  --overwrite_ckpt --wandb --no_check_link_existence --model GAT --version static > out_gat2 2> err_gat2 #& echo $! > run3.pid

# RGCN

python -u main.py --data_name darpa_0to125 --parallelism 8 --epochs 50 --batch 200 --save_dir RGCN_production_0to125_undirected --data_dir DATA --num_runs 1 --patience 5 --overwrite_ckpt --wandb --no_check_link_existence --model RGCN --version static > out_RGCN 2> err_RGCN #& echo $! > run2.pid
python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir RGCN_production_126to210_undirected --data_dir DATA --num_runs 1 --patience 5  --overwrite_ckpt --wandb --no_check_link_existence --model RGCN --version static > out_RGCN2 2> err_RGCN2 #& echo $! > run.pid

# GAT with TransR
# RGCN with TransR

#nohup python -u main.py --data_name darpa_0to125 --parallelism 8 --epochs 50 --batch 200 --save_dir MLP_production_0to125 --data_dir DATA --num_runs 1 --patience 5 --wandb --debug --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp 2> err_mlp &
#nohup python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir MLP_production_126to210 --data_dir DATA --num_runs 1 --patience 5 --wandb --debug --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp2 2> err_mlp2 &
#nohup python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir TGN_production_126to210_no_check --data_dir DATA --num_runs 1 --patience 5 --wandb --overwrite_ckpt --no_check_link_existence --debug > out_ 2> err_ &
#nohup python -u main.py --data_name darpa_0to125 --parallelism 8 --epochs 50 --batch 200 --save_dir MLP_production_0to125 --data_dir DATA --num_runs 1 --patience 5 --wandb --debug --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp 2> err_mlp &
