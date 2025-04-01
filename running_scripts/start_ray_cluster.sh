#!/bin/bash
PROJ_PATH="$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/"
# Get ray head node
HEAD_NODE=`cat $PBS_NODEFILE|head -n 1`
NODE_IFACE="hsn0"
HEAD_NODE_IP=`ip -4 addr show $NODE_IFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
WORKER_NODES=`cat $PBS_NODEFILE|tail -n +2`

echo "Head node: $HEAD_NODE, ip: $HEAD_NODE_IP"
# Connect to the head node and start the ray head
ssh "$HEAD_NODE" "cd $PROJ_PATH;source vllm_env_moe;source vllm_env_vars_ray;./start_ray.sh head $HEAD_NODE_IP $NODE_IFACE"
sleep 1

# Connect to the worker nodes and start the ray workers
for WORKER in $WORKER_NODES; do
    echo "Worker node: $WORKER"
    ssh "$WORKER" "cd $PROJ_PATH;source vllm_env_moe;source vllm_env_vars_ray;./start_ray.sh worker $HEAD_NODE_IP $NODE_IFACE"
done

wait