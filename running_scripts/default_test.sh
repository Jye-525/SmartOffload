#!/bin/bash

# Prompt-only test mode

# seq_lens = 1k, 2k, 4k, 8k, 16k, 32k
#prompt_lens=(127 255 511 1023 2047 4095 8191 16383 32767)
prompt_lens=(1023)
# prompt_len=8191
decode_len=1
# decode_lens=(128 256 512 1024 2048 4096 8192)
# decode_lens=(1023)
num_reqs=1
# 0: no offloading, 1: default offloading policy, 2: RingBuffer-OnDemand-case1(first k layers), 3: RingBuffer-OnDemand-case2(last k layers), 4: RingBuffer-Prefetch
offload_type=1 

model="mistralai/Mistral-7B-v0.3"
model_path="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
exec_path="${HOME}/moe_mix_precision/SmartOffload_polaris/benchmarks"
executor_backend="mp"
exec_mode="cudagraph" # eager or cudagraph

LOG_PATH="${HOME}/moe_mix_precision/SmartOffload_polaris/scripts/logs_eager_vs_cudagraph"
model_name=$(echo $model | cut -d'/' -f2)

[ -d $LOG_PATH ] || mkdir -p $LOG_PATH

if [ $exec_mode == "eager" ]; then
    USE_CUDA_GRAPH=0
elif [ $exec_mode == "cudagraph" ]; then
    USE_CUDA_GRAPH=1
else
    echo "Invalid execution mode"
    exit 1
fi

if [ $offload_type -eq 0 ]; then
    # without offloading
    for prompt_len in ${prompt_lens[@]}; do
    # for decode_len in ${decode_lens[@]}; do
        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ..."
        LOG_FILE_NAME="${LOG_PATH}/result_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_default_${exec_mode}.log"
        PROF_FILE_NAME="${LOG_PATH}/report_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_default_${exec_mode}.nsys-rep"
        base_command="python ${exec_path}/benchmark_throughput.py \
            --backend vllm \
            --input-len $prompt_len \
            --output-len $decode_len \
            --num-prompts $num_reqs \
            --model $model \
            --download-dir $model_path \
            --trust-remote-code \
            --distributed-executor-backend ${executor_backend} \
            --tensor-parallel-size 1"
        
        if [ $USE_CUDA_GRAPH -eq 0 ]; then
            base_command="${base_command} --enforce-eager"
        fi
        
        # eval "nsys profile --force-overwrite true -o ${PROF_FILE_NAME} --trace-fork-before-exec=true --cuda-graph-trace=node \
        #     ${base_command}" > ${LOG_FILE_NAME} 2>&1
        eval "${base_command}" > ${LOG_FILE_NAME} 2>&1
        
        sleep 2
        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
    done
elif [ $offload_type -eq 1 ]; then
    # with default offloading policy (gb=0.9)
    offload_gb=0.9
    for prompt_len in ${prompt_lens[@]}; do
    # for decode_len in ${decode_lens[@]}; do
        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ...offload_type=${offload_type}"
        LOG_FILE_NAME="${LOG_PATH}/result_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_default_offload_${offload_gb}gb_${exec_mode}.log"
        PROF_FILE_NAME="${LOG_PATH}/report_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_default_offload_${offload_gb}gb_${exec_mode}.nsys-rep"
        base_command="python ${exec_path}/benchmark_throughput.py \
            --backend vllm \
            --input-len $prompt_len \
            --output-len $decode_len \
            --num-prompts $num_reqs \
            --model $model \
            --download-dir $model_path \
            --trust-remote-code \
            --distributed-executor-backend ${executor_backend} \
            --tensor-parallel-size 1 \
            --cpu-offload-gb ${offload_gb}"

        if [ $USE_CUDA_GRAPH -eq 0 ]; then
            base_command="${base_command} --enforce-eager"
        fi

        eval "nsys profile --force-overwrite true -o ${PROF_FILE_NAME} --trace-fork-before-exec=true --cuda-graph-trace=node \
            ${base_command}" > ${LOG_FILE_NAME} 2>&1

        sleep 2
        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
    done
elif [ $offload_type -eq 2 ]; then
    # with Ringbuffer-Ondemand offloading policy (layers=2, type=all)
    # Try using cuda_graph to replace --enforce-eager
    offload_layers=2
    for prompt_len in ${prompt_lens[@]}; do
    # for decode_len in ${decode_lens[@]}; do
        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ...offload_type=${offload_type}"
        LOG_FILE_NAME="${LOG_PATH}/result_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand_${offload_layers}layers_all_${exec_mode}.log"
        PROF_FILE_NAME="${LOG_PATH}/report_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand_${offload_layers}layers_all_${exec_mode}.nsys-rep"
        base_command="python ${exec_path}/benchmark_throughput.py \
            --backend vllm \
            --input-len $prompt_len \
            --output-len $decode_len \
            --num-prompts $num_reqs \
            --model $model \
            --download-dir $model_path \
            --trust-remote-code \
            --distributed-executor-backend ${executor_backend} \
            --tensor-parallel-size 1 \
            --cpu-offload-layers ${offload_layers} \
            --cpu-offload-type all"

        if [ $USE_CUDA_GRAPH -eq 0 ]; then
            base_command="${base_command} --enforce-eager"
        fi

        eval "nsys profile --force-overwrite true -o ${PROF_FILE_NAME} --trace-fork-before-exec=true --cuda-graph-trace=node \
            ${base_command}" > ${LOG_FILE_NAME} 2>&1
        sleep 2
        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
    done
elif [ $offload_type -eq 3 ]; then
    # with Ringbuffer-Ondemand offloading policy (layers=2, type=all)
    offload_layers=2
    for prompt_len in ${prompt_lens[@]}; do
    # for decode_len in ${decode_lens[@]}; do
        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ...offload_type=${offload_type}"
        LOG_FILE_NAME="${LOG_PATH}/result_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand3_${offload_layers}layers_all_${exec_mode}.log"
        PROF_FILE_NAME="${LOG_PATH}/report_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand3_${offload_layers}layers_all_${exec_mode}.nsys-rep"
        base_command="python ${exec_path}/benchmark_throughput.py \
            --backend vllm \
            --input-len $prompt_len \
            --output-len $decode_len \
            --num-prompts $num_reqs \
            --model $model \
            --download-dir $model_path \
            --trust-remote-code \
            --distributed-executor-backend ${executor_backend} \
            --tensor-parallel-size 1 \
            --cpu-offload-layers ${offload_layers} \
            --cpu-offload-type all"

        if [ $USE_CUDA_GRAPH -eq 0 ]; then
            base_command="${base_command} --enforce-eager"
        fi

        eval "nsys profile --force-overwrite true -o ${PROF_FILE_NAME} --trace-fork-before-exec=true --cuda-graph-trace=node \
            ${base_command}" > ${LOG_FILE_NAME} 2>&1
        sleep 2
        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
    done
elif [ $offload_type -eq 4 ]; then
    # with Ringbuffer-Ondemand offloading policy (layers=2, type=all)
    offload_layers=4
    for prompt_len in ${prompt_lens[@]}; do
    # for decode_len in ${decode_lens[@]}; do
        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ...offload_type=${offload_type}"
        LOG_FILE_NAME="${LOG_PATH}/result_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand4_${offload_layers}layers_all.log"
        PROF_FILE_NAME="${LOG_PATH}/report_${model_name}_offload_p${prompt_len}_g${decode_len}_r${num_reqs}_tp1_pp1_ringbuffer_ondemand4_${offload_layers}layers_all.nsys-rep"
        base_command="python ${exec_path}/benchmark_throughput.py \
            --backend vllm \
            --input-len $prompt_len \
            --output-len $decode_len \
            --num-prompts $num_reqs \
            --model $model \
            --download-dir $model_path \
            --trust-remote-code \
            --distributed-executor-backend ${executor_backend} \
            --tensor-parallel-size 1 \
            --cpu-offload-layers ${offload_layers} \
            --cpu-offload-type all"

        if [ $USE_CUDA_GRAPH -eq 0 ]; then
            base_command="${base_command} --enforce-eager"
        fi

        eval "nsys profile --force-overwrite true -o ${PROF_FILE_NAME} --trace-fork-before-exec=true --cuda-graph-trace=node \
            ${base_command}" > ${LOG_FILE_NAME} 2>&1

        sleep 2
        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
    done
else
    echo "Invalid offload type"
fi
