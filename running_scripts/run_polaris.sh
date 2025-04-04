#!/bin/bash -l
#PBS -l select=10:system=polaris
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -A RECUP

PROJ_PATH=$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts
source $PROJ_PATH/vllm_env_moe
source $PROJ_PATH/vllm_env_vars_ray polaris

echo "Allocated Node lists...."
cat $PBS_NODEFILE

cd $PROJ_PATH
echo "Start running default_online_real.sh ... current dir: $(pwd)"
./default_online_real.sh
cd -
