#!/bin/bash

#SBATCH --job-name=<THE NAME OF YOUR JOB>
#SBATCH --nodes=<THE NUMBER OF NODES FOR YOUR EXPERIMENT>
#SBATCH --tasks-per-node=1
#SBATCH --output=slurm_%x_%A.out
#SBATCH --error=slurm_%x_%A.err

env=<THE NAME OF THE CONDA ENVIRONMENT>
export NUM_CPUS_PER_TASK=<THE NUMBER OF CPUS AVAILABLE FOR EACH MODEL CONFIG>
export NUM_GPUS_PER_TASK=<THE NUMBER/PERCENTAGE OF GPUS AVAILABLE FOR EACH MODEL CONFIG>

worker_num=<THE NUMBER OF NODES FOR YOUR EXPERIMENT - 1> # Must be one less that the total number of nodes

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
export redis_password

ulimit -n 65536 # increase limits to avoid to exceed redis connections

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done


# Jemalloc
# export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000
# export LD_PRELOAD=~/miniconda3/envs/$env/lib/libjemalloc.so

export LD_LIBRARY_PATH=~/miniconda3/envs/$env/lib

# Parallelization settings
# export KMP_SETTING=granularity=fine,compact,1,0                                       ## affinity 1
# #export KMP_SETTING=KMP_AFFINITY=noverbose,warnings,respect,granularity=core,none      ## affinity 2
# export OMP_NUM_THREADS=2
# export KMP_BLOCKTIME=50

echo MALLOC_CONF=$MALLOC_CONF
echo LD_PRELOAD=$LD_PRELOAD
echo OMP_THREADS=$OMP_NUM_THREADS
echo KMP_BLOCKTIME=$KMP_BLOCKTIME
echo KMP_SETTING=$KMP_SETTING

python -u main.py <YOUR ARGUMENTS>



