#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=05:00:00
#PBS -q preemptable
#PBS -A RECUP

source $HOME/moe_mix_precision/vllm_env_moe
source $HOME/moe_mix_precision/vllm_env_vars polaris

cd $HOME/moe_mix_precision/SmartOffload_polaris/scripts
echo "Start running default_test1.sh ... current dir: $(pwd)"
./default_test1.sh
cd -