#!/bin/bash


export NUM_GPUS_PER_TASK=0.5
python -u main.py --data_name darpa_trace_0to210 --model TGN --parallelism 2 --epochs 50 --batch 200 --save_dir big_X_TGN_trace --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_big_X_TGN_trace 2> err_big_X_TGN_trace
export NUM_GPUS_PER_TASK=0.3
python -u main.py --data_name darpa_theia_0to25 --model TGN --parallelism 3 --epochs 50 --batch 200 --save_dir big_X_TGN_theia --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_big_X_TGN_theia 2> err_big_X_TGN_theia
#python -u main.py --data_name darpa_trace_0to210 --model MLP --parallelism 2 --epochs 50 --batch 200 --save_dir timing --data_dir DATA --num_runs 1 --patience 5 --wandb --cpu --no_check_link_existence > out_timing_MLP 2> err_timing_MLP
#python -u main.py --data_name darpa_trace_0to210 --model RGCN --version static --parallelism 2 --epochs 50 --batch 200 --save_dir timing --cpu --num_runs 1 --patience 5 --wandb --no_check_link_existence > out_timing_RGCN 2> err_timing_RGCN
#python -u main.py --data_name darpa_trace_0to210 --model GAT --version static --parallelism 2 --epochs 50 --batch 200 --save_dir timing --cpu --num_runs 1 --patience 5 --wandb --no_check_link_existence > out_timing_GAT 2> err_timing_GAT

# debug
#python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 1 --batch 200 --save_dir TGN_debug --data_dir DATA --num_runs 1 --overwrite_ckpt --patience 5 --wandb --no_check_link_existence --debug

#export NUM_GPUS_PER_TASK=0.1
# Temporal Baseline
#export NUM_GPUS_PER_TASK=0.33
# Temporal Baseline
#python -u main.py --data_name darpa_trace_0to210 --parallelism 10 --epochs 50 --batch 200 --save_dir MLP_production_trace_0to210 --data_dir DATA --num_runs 5 --patience 5 --wandb --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp_trace 2> err_mlp_trace

#python -u main.py --data_name darpa_theia_0to25 --parallelism 10 --epochs 50 --batch 200 --save_dir MLP_production_theia_0to125 --data_dir DATA --num_runs 5 --patience 5 --wandb --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp 2> err_mlp
#nohup python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir MLP_production_126to210 --data_dir DATA --num_runs 1 --patience 5 --wandb --debug --overwrite_ckpt --no_check_link_existence --model MLP > out_mlp2 2> err_mlp2 &
#nohup python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir TGN_production_126to210_no_check --data_dir DATA --num_runs 1 --patience 5 --wandb --overwrite_ckpt --no_check_link_existence --debug > out_ 2> err_ &

#export NUM_GPUS_PER_TASK=0.33
#python -u main.py --data_name darpa_theia_0to25 --parallelism 3 --epochs 50 --batch 200 --save_dir TGN_hetero_and_layers_theia_0to25 --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_hetero 2> err_hetero

#export NUM_GPUS_PER_TASK=1
#python -u main.py --data_name darpa_trace_0to210 --parallelism 1 --epochs 50 --batch 200 --save_dir TGN_memory2_experiment_trace_0to210 --data_dir DATA --num_runs 5 --patience 5 --overwrite_ckpt --wandb --no_check_link_existence > out_memory 2> err_memory
# TGN 
#python -u main.py --data_name darpa_0to125 --parallelism 1 --epochs 50 --batch 200 --save_dir TGN_production_0to125_debug_features --data_dir DATA --num_runs 1 --patience 5 --wandb --no_check_link_existence >out2 2>err2

#python -u main.py --data_name darpa_116to125 --parallelism 2 --epochs 2 --batch 200 --save_dir TGN_debug --data_dir DATA --num_runs 4 --patience 5 --wandb --no_check_link_existence --overwrite_ckpt > out_parallel 2> err_parallel
#export NUM_GPUS_PER_TASK=0.25
#python -u main.py --data_name darpa_theia_0to25 --parallelism 4 --epochs 50 --batch 200 --save_dir TGN_memory_experiment_theia_0to25 --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_debug 2> err_debug
#export NUM_GPUS_PER_TASK=0.5
#python -u main.py --data_name darpa_trace_0to210 --parallelism 2 --epochs 50 --batch 200 --save_dir TGN_memory_experiment_trace_0to210 --data_dir DATA --num_runs 5 --patience 5 --wandb --no_check_link_existence > out_debug2 2> err_debug2"""
#nohup python -u main.py --data_name darpa_126to210 --parallelism 8 --epochs 50 --batch 200 --save_dir TGN_production_126to210_no_check --data_dir DATA --num_runs 1 --patience 5 --wandb --no_check_link_existence --debug
#python -u main.py --data_name darpa_0to125 --parallelism 8 --epochs 50 --batch 200 --save_dir TGN_production_0to125 --data_dir DATA --num_runs 1  --patience 5 --wandb --debug

# CLUSTER
#export NUM_GPUS_PER_TASK=0
#python -u main.py --data_name darpa_theia_0to25 --model TGN --parallelism 100 --epochs 50 --batch 200 --save_dir TGN_production_darpa_theia_0to25_big --num_runs 5 --patience 5 --wandb --no_check_link_existence --cluster > out 2> err
export NUM_GPUS_PER_TASK=0
#python -u main.py --data_name darpa_theia_0to25 --model RGCN --version static --parallelism 40 --epochs 50 --batch 200 --save_dir final_experiments_darpa_theia_0to25 --num_runs 5 --patience 5 --wandb --no_check_link_existence --cluster > out_RGCN_theia 2> err_RGCN_theia
#python -u main.py --data_name darpa_theia_0to25 --model GAT --version static --parallelism 40 --epochs 50 --batch 200 --save_dir final_experiments_darpa_theia_0to25 --num_runs 5 --patience 5 --wandb --no_check_link_existence --cluster > out_GAT_theia 2> err_GAT_theia
#python -u main.py --data_name darpa_trace_0to210 --model RGCN --version static --parallelism 40 --epochs 50 --batch 200 --save_dir final_experiments_darpa_trace_0to210 --num_runs 5 --patience 5 --wandb --no_check_link_existence --cluster > out_RGCN_trace 2> err_RGCN_trace
#python -u main.py --data_name darpa_trace_0to210 --model GAT --version static --parallelism 40 --epochs 50 --batch 200 --save_dir final_experiments_darpa_trace_0to210 --num_runs 5 --patience 5 --wandb --no_check_link_existence --cluster > out_GAT_trace 2> err_GAT_trace
