#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=15:00:00
#PBS -q preemptable
#PBS -A VeloC

PROJ_PATH=$HOME/moe_mix_precision/running_scripts/
source $PROJ_PATH/vllm_env_moe
source $PROJ_PATH/vllm_env_vars_ray polaris

echo "Allocated Node lists...."
cat $PBS_NODEFILE

# vLLM native offloading
cd $PROJ_PATH
### w/o offloading
echo "Start running default_online_real_v84_1.sh ... current dir: $(pwd)"
./default_online_real_v84_1.sh "default"
sleep 5
# ./default_online_real_v84_1.sh "kv-usage-aware"
# sleep 5
# ./default_online_real_v84_1.sh "kv-usage-aware-1"
# sleep 5
./default_online_real_v84_1.sh "evict-optimal-7"
cd -
