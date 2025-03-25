#!/bin/bash
# Check for minimum number of required arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 head|worker ray_head_node_address if_namer"
    exit 1
fi

RAY_NODE_TYPE="$1"
RAY_HEAD_NODE_ADDRESS="$2"
NODE_IFACE=$3 # Should be head or worker

# Validate node type
if [ "${RAY_NODE_TYPE}" != "head" ] && [ "${RAY_NODE_TYPE}" != "worker" ]; then
    echo "Error: Node type must be head or worker"
    exit 1
fi

# get the IP address of the node with a given interface
RAY_NODE_ADDRESS=`ip -4 addr show $NODE_IFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}'`
NUM_CPUS=64
NUM_GPUS=4

# Command setup for head or worker node
RAY_START_CMD="ray start --num-cpus $NUM_CPUS --num-gpus $NUM_GPUS"
if [ "${RAY_NODE_TYPE}" == "head" ]; then
    RAY_START_CMD+=" --head --node-ip-address=$RAY_NODE_ADDRESS --port=6379 --disable-usage-stats"
else
    RAY_START_CMD+=" --address=${RAY_HEAD_NODE_ADDRESS}:6379 --node-ip-address=$RAY_NODE_ADDRESS --disable-usage-stats"
fi

# Run the ray start command
ray_path=`which ray`
echo "ray_path: $ray_path"
echo "Running command: $RAY_START_CMD"
$RAY_START_CMD

sleep 5
