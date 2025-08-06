#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=20:00:00
#PBS -q preemptable
#PBS -A RECUP

PROJ_PATH=$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/
source $PROJ_PATH/vllm_env_moe
source $PROJ_PATH/vllm_env_vars_ray polaris

echo "Allocated Node lists...."
cat $PBS_NODEFILE

cd $PROJ_PATH
echo "Start running default_online_real_v84.sh ... current dir: $(pwd)"
### w/o offloading
./default_online_real_v84.sh 0 0 0 "offline" "mp"
# ### smart offloading
# total_layers=32 # per GPU
# # This corresponding to different K interval for llama model
# #k_values=(32 16 8 6 4 2 1)
# k_values=(1)
# # ofd_layers=(1 2 4 6 8 16 32)
# #k=32 down to 1

# for k in ${k_values[@]}; do
#     echo "Start running default_online_real_v84.sh vLLM smart offloading - offload interval ${k} ... current dir: $(pwd)"
#     ./default_online_real_v84.sh 2 $k 0 "offline" "mp"
#     sleep 5
# done
cd -
