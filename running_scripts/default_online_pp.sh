#!/bin/bash
PROJ_PATH="$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/"

source $PROJ_PATH/vllm_env_vars_ray

echo "Start running default_online_pp.sh ..."

num_reqs=(64 128)
offload_type=0 # 0: no offloading, 1: smart_offload
#  "prompt_only" "decode_only"  "prompt_decode"
test_cases=("prompt_only")

PP=4
TP=1
model="deepseek-ai/deepseek-coder-33b-base"
#model="deepseek-ai/deepseek-coder-33b-large"
model_path="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
exec_path="${HOME}/moe_mix_precision/SmartOffload_polaris/benchmarks"
executor_backend="ray" # "ray" or "mp", for "mp", it only supports on a single node (PP * TP <= 4)
exec_mode="eager"
USE_PROFILING=0

model_name=$(echo $model | cut -d'/' -f2)
LOG_PATH="${HOME}/moe_mix_precision/SmartOffload_polaris/running_scripts/logs_${model_name}_tp${TP}_pp${PP}"

[ -d $LOG_PATH ] || mkdir -p $LOG_PATH

#################################################################################
start_ray_cluster() {
    RAY_SCRIPT="$PROJ_PATH/start_ray_cluster.sh"
    eval "$RAY_SCRIPT"
    sleep 10
    echo "Ray cluster started ..."
    eval "ray status"
}

stop_ray_cluster() {
    eval "ray stop"
    sleep 5
    eval "ray stop"
    echo "Ray cluster stopped ..."
}

start_vllm_server() {
    num_req=$1
    log_file=$2
    profile_file=$3
    # max_num_seqs=$(( num_req > 256 ? 256 : num_req ))
    max_num_seqs=256
    vllm_cmd="vllm serve ${model} \
        --download-dir ${model_path} \
        --trust-remote-code \
        --enforce-eager \
        --distributed-executor-backend ${executor_backend} \
        --tensor-parallel-size ${TP} \
        --pipeline-parallel-size ${PP} \
        --disable-log-requests \
        --max-num-seqs ${max_num_seqs} \
        --enable-chunked-prefill=False \
        --collect-layer-fwd-time "

    eval "$vllm_cmd" > "$log_file" 2>&1 &
    sleep_inter=5
    sleep $sleep_inter
    echo "Start checking of the vLLM server start successfully ..."
    total_wait_time=300
    waiting_time=0
    while true; do
        if [ $waiting_time -ge $total_wait_time ]; then
            echo "vLLM server failed to start ..."
            break
        fi
        # Check for the Uvicorn message
        if grep -q "INFO:     Uvicorn running on http" "$log_file"; then
            echo "vLLM server started successfully!"
            break
        fi
        sleep $sleep_inter
        waiting_time=$((waiting_time + sleep_inter))
    done
}

stop_vllm_server() {
    killall -9 /home/jieye/venvs/vllm_moe/bin/python
    sleep 5
    killall -9 /home/jieye/venvs/vllm_moe/bin/python 
    echo "VLLM server stopped ..."
}


echo "Start running without offloading ..."
if [ $offload_type -eq 0 ]; then
    echo "Start running without offloading ... using test case ${test_cases}"
    for test_case in "${test_cases[@]}"; do
        echo "Start running without offloading ... using test case ${test_case}"
        if [ "$test_case" = "prompt_only" ]; then
            # prompt_lens=(511 1023 2047 4095 8191)
            prompt_lens=(1023)
            decode_len=1
            for num_req in ${num_reqs[@]}; do
                for prompt_len in ${prompt_lens[@]}; do
                    echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ..."
                    SERVER_LOG_FILE_NAME="${LOG_PATH}/server_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.log"
                    PROF_FILE_NAME="${LOG_PATH}/server_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.nsys-rep"

                    if [ $executor_backend = "ray" ]; then
                        # stop ray cluster in case it is not stopped by previous run
                        stop_ray_cluster
                        # start ray cluster
                        start_ray_cluster
                    fi
                    # Start vLLM server
                    start_vllm_server ${num_req} ${SERVER_LOG_FILE_NAME} ${PROF_FILE_NAME}
                    sleep 2
                    client_cmd="python ${exec_path}/benchmark_serving.py \
                        --backend vllm \
                        --model $model \
                        --dataset-name fixed-len \
                        --fixed-input-len $prompt_len \
                        --fixed-output-len $decode_len \
                        --ignore-eos \
                        --num-prompts $num_req "

                    CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.log"
                    eval "${client_cmd}" > ${CLIENT_LOG_FILE_NAME} 2>&1
                    sleep 10
                    stop_vllm_server
                    if [ $executor_backend = "ray" ]; then
                        stop_ray_cluster
                    fi
                    echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
                done
            done
        elif [ "$test_case" = "prompt_decode" ]; then
            echo "I am here. Start running without offloading ... using test case ${test_case}"
            # prompt_lens=(511 2047 8191)
            # prompt_lens=(511 1023 2047 4095 8191)
            # prompt_lens=(511 1023 2047 4095 8191)
            prompt_lens=(1023)
            decode_lens=(128)
            for num_req in ${num_reqs[@]}; do
                for prompt_len in ${prompt_lens[@]}; do
                    for decode_len in ${decode_lens[@]}; do 
                        echo "Start with prompt length ${prompt_len} and decode length ${decode_len} ..."
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/server_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.log"
                        PROF_FILE_NAME="${LOG_PATH}/server_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.nsys-rep"

                        if [ $executor_backend = "ray" ]; then
                            stop_ray_cluster
                            start_ray_cluster
                        fi

                        start_vllm_server ${num_req} ${SERVER_LOG_FILE_NAME} ${PROF_FILE_NAME}
                        sleep 2
                        echo "Start to Serve ...."
                        client_cmd="python ${exec_path}/benchmark_serving.py \
                            --backend vllm \
                            --model $model \
                            --dataset-name fixed-len \
                            --fixed-input-len $prompt_len \
                            --fixed-output-len $decode_len \
                            --ignore-eos \
                            --num-prompts $num_req "

                        CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_${model_name}_fixed_p${prompt_len}_g${decode_len}_r${num_req}_tp${TP}_pp${PP}_${exec_mode}.log"
                        eval "${client_cmd}" > ${CLIENT_LOG_FILE_NAME} 2>&1
                        sleep 10
                        stop_vllm_server
                        if [ $executor_backend = "ray" ]; then
                            stop_ray_cluster
                        fi
                        echo "Done with prompt length ${prompt_len} and decode length ${decode_len} ..."
                    done
                done
            done
        fi
    done
fi
