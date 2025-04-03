#!/opt/homebrew/bin/bash
### Note the bash version should be 4.0 or above
PROJ_PATH="$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/"

source $PROJ_PATH/vllm_env_vars_ray

echo "Start running default_online_pp.sh ..."

PYTHON_PATH=`which python`
echo "The current python executable path is $PYTHON_PATH"
################### Configurable Parameters to Change #####################
EXEC_PATH="$PROJ_PATH/../benchmarks/"
MODEL_PATH="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
BASE_DATASET_PATH="/lus/eagle/projects/RECUP/jye/datasets/"
LOG_BASE_PATH="${HOME}/moe_mix_precision/SC25_logs/"

# Model configurations
declare -A MODEL_CONFIG=(
    # Format: "TP PP MAX_MODEL_LEN GPU_MEMORY_LIMIT"
    ["deepseek-ai/deepseek-coder-33b-base"]="4 1 32768 0.8"
    ["meta-llama/Llama-3.3-70B-Instruct"]="4 2 32768 0.8"
    ["alpindale/goliath-120b"]="4 4 4096 0.8"
    ["meta-llama/Llama-3.1-405B"]="4 10 32768 0.8"
)

TESTA_CASES=("prompt-only") #  "prompt-only" "decode-only"  "prompt-decode"
NUM_TRIES=2
NUM_REQS=(10)

# Associative array declaration for different datasets
SUBTASKS="hotpotqa,2wikimqa" # used for longbench
declare -A DATASETS=(
    ["longbench"]="--dataset-name longbench --dataset-path ${BASE_DATASET_PATH} --longbench-subtasks \"${SUBTASKS}\""
    ["gsm8k"]="--dataset-name gsm8k --dataset-path ${BASE_DATASET_PATH}"
    ["sharedgpt"]="--dataset-name sharedgpt --dataset-path ${BASE_DATASET_PATH}"
)

# Associative array declaration for synthetic datasets
INPUT_LENS=(8191)
OUT_LENS=(1) 
declare -A SYNT_DATASETS=(
    ["random"]="--dataset-name random --random-input-len __INPUT_LEN__ --random-output-len __OUT_LEN__ --random-range-ratio 1.0"
    ["fixed-len"]="--dataset-name fixed-len --fixed-input-len __INPUT_LEN__ --fixed-output-len __OUT_LEN__"
)

DATASET_NAME="longbench"
MODEL="deepseek-ai/deepseek-coder-33b-base"
# Model="meta-llama/Llama-3.3-70B-Instruct"
# Model="alpindale/goliath-120b"
# Model="meta-llama/Llama-3.1-405B"
IFS=' ' read -r TP PP MAX_MODEL_LEN GPU_MEM_LIMIT <<< "${MODEL_CONFIG[$MODEL]}"

OFFLOAD_TYPE=0 # 0 no offloading, 1: vllm naive offloading, 2: smart_offload
OFFLOAD_LAYERS="1,4,7" # used for smart_offload
OFFLOAG_GB=6 # used for vllm naive offloading
declare -A OFFLOAD_CONFIG=(
    # Format: "OFFLOAD_TYPE "
    ["1"]="--cpu-offload-method default --cpu-offload-gb ${OFFLOAG_GB}"
    ["2"]="--cpu-offload-method smart_offload --cpu-offload-layers \"${OFFLOAD_LAYERS}\" --param-offload-target all"
)


EXECUTOR_BACKEND="mp" # "ray" or "mp", for "mp", it only supports on a single node (PP * TP <= 4)
#EXEC_MODE="eager" # "eager"
LOG_STATS_INTER=1 # in seconds
PREEMPTION_MODE="swap" # "recompute" or "swap"

MODEL_NAME=$(echo $MODEL | cut -d'/' -f2)
LOG_PATH="${LOG_BASE_PATH}/logs_${MODEL_NAME}_tp${TP}_pp${PP}"
[ -d $LOG_PATH ] || mkdir -p $LOG_PATH

###################################### Related Helper functions #############################################
###################################### Start and Stop Ray Clsuter ###########################################
start_ray_cluster() {
    RAY_SCRIPT="$PROJ_PATH/start_ray_cluster.sh"
    eval "$RAY_SCRIPT"
    sleep 10
    echo "Ray cluster started ..."
    eval "ray status"
}

stop_ray_cluster() {
    RAY_STOP_SCRIPT="$PROJ_PATH/stop_ray_cluster.sh" 
    eval "$RAY_STOP_SCRIPT"
    sleep 5
    eval "$RAY_STOP_SCRIPT"
    echo "Ray cluster stopped ..."
}

###################################### Start and Stop vLLM Server ###########################################
start_vllm_server() {
    server_log_file=$1

    vllm_cmd="vllm serve ${MODEL} \
        --download-dir ${MODEL_PATH} \
        --trust-remote-code \
        --enforce-eager \
        --distributed-executor-backend ${EXECUTOR_BACKEND} \
        --tensor-parallel-size ${TP} \
        --pipeline-parallel-size ${PP} \
        --disable-log-requests \
        --enable-chunked-prefill=False \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEM_LIMIT} \
        --log-stats-interval ${LOG_STATS_INTER} \
        --preemption-mode ${PREEMPTION_MODE} \
        --collect-layer-fwd-time "

    if [ $OFFLOAD_TYPE -ne 0 ]; then
        vllm_cmd+="${OFFLOAD_CONFIG[$OFFLOAD_TYPE]}"
    fi
    
    eval "$vllm_cmd" > "$server_log_file" 2>&1 &
}

check_vllm_server_start() {
    server_log_file=$1
    sleep_inter=5
    total_wait_time=600
    
    sleep $sleep_inter
    echo "Start checking if the vLLM server started successfully..."
    
    waiting_time=0
    while true; do
        if [ $waiting_time -ge $total_wait_time ]; then
            echo "vLLM server failed to start ..."
            return 1
        fi
        # Check for the Uvicorn message
        if grep -q "INFO:     Uvicorn running on http" "$server_log_file"; then
            echo "vLLM server started successfully!"
            return 0
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

####################################### Run the client benchmark ##################################################
Run_client_bench() {
    local dataset_type=$1
    local client_log_file_name=$2
    local num_req=$3
    local extra_params=$4

    if [ "$dataset_type" = "synthetic" ]; then
        dataset_config="${extra_params}"
    elif [ "$dataset_type" = "real" ]; then
        dataset_config="${DATASETS[$DATASET_NAME]}" 
        gen_len=$extra_params 
        if [ $gen_len -gt 0 ]; then
            case "$DATASET_NAME" in
                "longbench")   dataset_config+=" --longbench-output-len $gen_len" ;;
                "gsm8k")       dataset_config+=" --gsm8k-output-len $gen_len" ;;
                "sharedgpt")   dataset_config+=" --sharedgpt-output-len $gen_len" ;;
            esac
        fi
    fi
    
    client_cmd="python ${EXEC_PATH}/benchmark_serving.py \
                    --backend vllm \
                    --model $MODEL \
                    --ignore-eos \
                    --num-prompts $num_req \
                    ${dataset_config} "

            
    eval "${client_cmd}" > "${client_log_file_name}" 2>&1
    sleep 180 # give the server some time to finish the requests
}

################################################## Run the Client Test ##########################################################
benchmark_with_real_dataset() {
    local gen_len=$1
    for num_req in ${NUM_REQS[@]}; do
        echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME} ..."
        for try_idx in $(seq 1 $NUM_TRIES); do
            if [ $gen_len -gt 0 ]; then
                SERVER_LOG_FILE_NAME="${LOG_PATH}/server_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_g${gen_len}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"
                CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_g${gen_len}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"
            else
                SERVER_LOG_FILE_NAME="${LOG_PATH}/server_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"
                CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"
            fi

            if [ $EXECUTOR_BACKEND = "ray" ]; then
                # stop ray cluster in case it is not stopped by previous run
                stop_ray_cluster
                # start ray cluster
                start_ray_cluster
            fi

            # Start vLLM server
            start_vllm_server ${SERVER_LOG_FILE_NAME}
            # Check vLLM server status
            check_vllm_server_start ${SERVER_LOG_FILE_NAME}
            if [ $? -ne 0 ]; then
                echo "Failed to start vLLM server. Continue to next test...."
                continue
            fi
            sleep 2

            # Start the client benchmark
            Run_client_bench "real" ${CLIENT_LOG_FILE_NAME} ${num_req} ${gen_len} 
            sleep 2

            # Stop the vLLM server
            stop_vllm_server

            # Stop the ray cluster if it was started
            if [ $EXECUTOR_BACKEND = "ray" ]; then
                stop_ray_cluster
            fi
        done
        echo "Finished inference with num_req=${num_req} requests using dataset ${DATASET_NAME} ..."
    done 
}

benchmark_with_synthetic_dataset() {
    local test_case=$1
    if [ "$test_case" = "prompt-only" ]; then
        OUT_LENS=(1) # decode length
    fi

    for num_req in ${NUM_REQS[@]}; do
        echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME} ..."
        for input_len in ${INPUT_LENS[@]}; do
            for out_len in ${OUT_LENS[@]}; do
                # Replace placeholders with actual values
                dataset_config="${SYNT_DATASETS[$DATASET_NAME]}"
                dataset_config="${dataset_config//__INPUT_LEN__/$input_len}"
                dataset_config="${dataset_config//__OUT_LEN__/$out_len}"
                echo "Dataset: $DATASET_NAME, Config: $dataset_config"

                for try_idx in $(seq 1 $NUM_TRIES); do
                    SERVER_LOG_FILE_NAME="${LOG_PATH}/server_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_p${input_len}_g${out_len}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"
                    CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_${MODEL_NAME}_d${DATASET_NAME}_c${MAX_MODEL_LEN}_p${input_len}_g${out_len}_r${num_req}_tp${TP}_pp${PP}_gpu${GPU_MEM_LIMIT}_eager_${try_idx}.log"

                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        # stop ray cluster in case it is not stopped by previous run
                        stop_ray_cluster
                        # start ray cluster
                        start_ray_cluster
                    fi

                    # Start vLLM server
                    start_vllm_server ${SERVER_LOG_FILE_NAME}
                    # Check vLLM server status
                    check_vllm_server_start ${SERVER_LOG_FILE_NAME}
                    if [ $? -ne 0 ]; then
                        echo "Failed to start vLLM server. Continue to next test...."
                        continue
                    fi
                    sleep 2

                    # Start the client benchmark
                    Run_client_bench "synthetic" ${CLIENT_LOG_FILE_NAME} ${num_req} "${dataset_config}" 
                    sleep 2

                    # Stop the vLLM server
                    stop_vllm_server

                    # Stop the ray cluster if it was started
                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        stop_ray_cluster
                    fi
                done
            done
        done
    done
}

#################################################################################


if [ $OFFLOAD_TYPE -eq 0 ]; then
    # gen_len=-1
    echo "Running without offloading ..."
    for test_case in "${TESTA_CASES[@]}"; do
        if [ "$test_case" = "prompt-only" ]; then
            gen_len=1 # decode length
        elif [ "$test_case" = "prompt-decode" ]; then
            gen_len=-1 # decode length, meaning that we did not override the decode length
        else
            echo "Invalid test case ${test_case}. Continue next..."
            continue
        fi
        echo "gen_len=" $gen_len
        case "$DATASET_NAME" in
            "longbench"|"gsm8k"|"sharedgpt")   benchmark_with_real_dataset $gen_len ;;
            "random"|"fixed-len")      benchmark_with_synthetic_dataset $test_case ;;
        esac  
    done 

elif [ $OFFLOAD_TYPE -eq 1 ]; then
    echo "Running experiments with vllm naive offloading ..."
    for test_case in "${TESTA_CASES[@]}"; do
        if [ "$test_case" = "prompt-only" ]; then
            gen_len=1 # decode length
        elif [ "$test_case" = "prompt-decode" ]; then
            gen_len=-1 # decode length, meaning that we did not override the decode length
        else
            echo "Invalid test case ${test_case}. Continue next..."
            continue
        fi
        case "$DATASET_NAME" in
            "longbench"|"gsm8k"|"sharedgpt")   benchmark_with_real_dataset $gen_len ;;
            "random"|"fixed-len")      benchmark_with_synthetic_dataset $test_case ;;
        esac  
    done 
elif [ $OFFLOAD_TYPE -eq 2 ]; then
    echo "Running experiments with our offloading approach ..."
    for test_case in "${TESTA_CASES[@]}"; do
        if [ "$test_case" = "prompt-only" ]; then
            gen_len=1 # decode length
        elif [ "$test_case" = "prompt-decode" ]; then
            gen_len=-1 # decode length, meaning that we did not override the decode length
        else
            echo "Invalid test case ${test_case}. Continue next..."
            continue
        fi
        case "$DATASET_NAME" in
            "longbench"|"gsm8k"|"sharedgpt")   benchmark_with_real_dataset $gen_len ;;
            "random"|"fixed-len")      benchmark_with_synthetic_dataset $test_case ;;
        esac  
    done 
else
    echo "Invalid offload type ${OFFLOAD_TYPE}. Exiting..."
    exit 1
fi
