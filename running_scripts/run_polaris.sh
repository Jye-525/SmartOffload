#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=05:00:00
#PBS -q preemptable
#PBS -A RECUP

source $HOME/moe_mix_precision/vllm_env_moe
source $HOME/moe_mix_precision/vllm_env_vars_ray polaris

cd $HOME/moe_mix_precision/SmartOffload_polaris/running_scripts
echo "Start running default_online_pp.sh ... current dir: $(pwd)"
./default_online_pp.sh
cd -