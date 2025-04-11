#!/bin/bash
PROJ_PATH="$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/"
#PROJ_PATH="$HOME/moe_benchmark/SmartOffload/running_scripts/"
# Get ray head node
HEAD_NODE=`cat $PBS_NODEFILE|head -n 1`
NODE_IFACE="hsn0" # for sophia, use 'bond0.2245'; for polaris use 'hsn0'
HEAD_NODE_IP=`ip -4 addr show $NODE_IFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
WORKER_NODES=`cat $PBS_NODEFILE|tail -n +2`

echo "Stopping Head node: $HEAD_NODE, ip: $HEAD_NODE_IP"
# Connect to the head node and stop the ray head
ssh "$HEAD_NODE" "cd $PROJ_PATH;source vllm_env_moe;source vllm_env_vars_ray;ray stop;rm -rf /tmp/vllm_jye/vllm_ray/*;rm -rf /tmp/vllm_jye/vllm_ipc/*"
#ssh "$HEAD_NODE" "cd $PROJ_PATH;source sophia_env;source vllm_env_vars_ray sophia;ray stop"
sleep 1

# Connect to the worker nodes and start the ray workers
for WORKER in $WORKER_NODES; do
    echo "Stopping Worker node: $WORKER"
    ssh "$WORKER" "cd $PROJ_PATH;source vllm_env_moe;source vllm_env_vars_ray;ray stop;rm -rf /tmp/vllm_jye/vllm_ray/*;rm -rf /tmp/vllm_jye/vllm_ipc/*" &
    #ssh "$WORKER" "cd $PROJ_PATH;source sophia_env;source vllm_env_vars_ray sophia;ray stop" &
done

wait
echo "Ray cluster stopped ..."
