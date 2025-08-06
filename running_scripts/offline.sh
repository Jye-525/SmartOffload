#!/bin/bash

EXE_FILE="$HOME/moe_mix_precision/avinash_updated/benchmarks/benchmark_throughput.py"
# meta-llama/Llama-3.1-8B,Qwen/Qwen1.5-MoE-A2.7B 
MODEL="meta-llama/Llama-3.1-8B"
MODEL_DWN_DIR="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
# MODEL="Qwen/Qwen3-14B"
# MODEL_DWN_DIR="/lus/grand/projects/VeloC/jye/viper2/huggingface-hub/"
CLS="vllm.v1.core.sched.scheduler.Scheduler"
EXEC_BACKEND="mp"
BATCHED_TOKENS=8192
MAX_MODEL_LEN=8192

# Set the environment variables
source $HOME/moe_benchmark/SmartOffload/running_scripts/vllm_env_vars_ray
export VLLM_USE_SMART_OFFLOADING=1
export VLLM_SMART_OFFLOAD_KVCACHE=0
export VLLM_USE_SMART_OFFLOADING_K=4
export VLLM_ENABLE_LAYER_FWD_TIMING=1

PROFILE_OUTPUT_DIR="/lus/eagle/projects/RECUP/jye/sophia_profile/"
# Create the output directory if it doesn't exist
if [ ! -d $PROFILE_OUTPUT_DIR ]; then
    mkdir -p $PROFILE_OUTPUT_DIR
fi

# Set the output file name
if [ $VLLM_USE_SMART_OFFLOADING == "1" ]; then
    PROFILE_OUTPUT_FILE="report-w_smart_bt${BATCHED_TOKENS}_int${VLLM_USE_SMART_OFFLOADING_K}.nsys-rep"
else
    PROFILE_OUTPUT_FILE="report-wo_smart_bt${BATCHED_TOKENS}.nsys-rep"
fi

INPUT_LEN=2048
OUTPUT_LEN=100
NUM_PROMPTS=200
# Run the benchmark
vllm_cmd="python3 $EXE_FILE --backend vllm \
--model $MODEL --download-dir $MODEL_DWN_DIR --trust-remote-code \
--tensor-parallel-size 1 --pipeline-parallel-size 1 --scheduler-cls $CLS --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --max-num-batched-tokens $BATCHED_TOKENS \
--input-len $INPUT_LEN --output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS \
--no-enable-prefix-caching --gpu-memory-utilization 0.9 \
--enforce-eager --distributed-executor-backend $EXEC_BACKEND"

nsys_cmd="nsys profile -o $PROFILE_OUTPUT_DIR/$PROFILE_OUTPUT_FILE \
--force-overwrite=true -t cuda,cudnn,cublas,nvtx \
--trace-fork-before-exec=true --cuda-flush-interval 1 "

# Combine the commands
# combined_cmd="$nsys_cmd $vllm_cmd"
combined_cmd="$vllm_cmd"
echo "Running command: $combined_cmd"
# Execute the command
eval $combined_cmd
# Wait for the command to finish
wait $!
echo "Command finished."