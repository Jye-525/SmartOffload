#!/bin/bash
### Note the bash version should be 4.0 or above
PROJ_PATH="$HOME/moe_mix_precision/SmartOffload_polaris/running_scripts/"
source $PROJ_PATH/vllm_env_vars_ray
PYTHON_PATH=`which python`
export VLLM_V1_SCHEDULE_METHOD="kv-usage-aware-1"
echo "The current python executable path is $PYTHON_PATH"
################### Configurable Parameters to Change #####################
EXEC_PATH="$PROJ_PATH/../benchmarks/"
MODEL_PATH="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
BASE_DATASET_PATH="/lus/eagle/projects/RECUP/jye/huggingface-hub/"
LOG_BASE_PATH="${HOME}/moe_mix_precision/hipc25_sched/"
# LOG_BASE_PATH="/lus/grand/projects/VeloC/jye/middleware_vllm_default/"

TESTA_CASES=("prompt-decode") #  "prompt-only" "decode-only"  "prompt-decode"
S_IDX=1
NUM_TRIES=1
# NUM_TRIES=1 # number of tries for each test case
# multi-news, gov_report (200), sharegpt (5000), lcc (500)
NUM_REQS=(5000) # for sharegpt
# NUM_REQS=(200) # for multi-news and gov_report 

DATASET_NAME="sharegpt" # longbench, gsm8k, sharegpt, random, fixed-len
MODEL="meta-llama/Llama-3.1-8B"
# MODEL="TroyDoesAI/Llama-3.1-13B-Instruct"
# MODEL="deepseek-ai/deepseek-coder-33b-base"
#MODEL="meta-llama/Llama-3.3-70B-Instruct"
# MODEL="alpindale/goliath-120b"
# MODEL="meta-llama/Llama-3.1-405B"

# Model configurations
gpu_mem_utils=(0.8)
# gpu_mem_utils=(0.9) # GPU memory utilization
declare -A MODEL_CONFIG=(
    # Format: "TP PP MAX_MODEL_LEN"
    ["meta-llama/Llama-3.1-8B"]="1 1 32768"
    ["TroyDoesAI/Llama-3.1-13B-Instruct"]="2 1 32768"
    ["deepseek-ai/deepseek-coder-33b-base"]="4 1 32768"
    ["meta-llama/Llama-3.3-70B-Instruct"]="4 2 32768"
    ["alpindale/goliath-120b"]="4 4 4096"
    ["meta-llama/Llama-3.1-405B"]="4 10 32768"
)
IFS=' ' read -r TP PP MAX_MODEL_LEN <<< "${MODEL_CONFIG[$MODEL]}"

# Associative array declaration for different datasets
SUBTASK="gov_report" # used for longbench, choices: gov_report, multi_news, lcc (500)
declare -A DATASETS=(
    ["longbench"]="--dataset-name longbench --dataset-path ${BASE_DATASET_PATH} --longbench-subtask \"${SUBTASK}\""
    ["gsm8k"]="--dataset-name gsm8k --dataset-path ${BASE_DATASET_PATH}"
    ["sharegpt"]="--dataset-name sharegpt --dataset-path \"${BASE_DATASET_PATH}/ShareGPT_V3_unfiltered_cleaned_split.json\""
)

# Associative array declaration for synthetic datasets
# NOTE: len(INPUT_LENS) == len(OUT_LENS)
INPUT_LENS=(1024)
OUT_LENS=(27)
declare -A SYNT_DATASETS=(
    ["random"]="--dataset-name random --random-input-len __INPUT_LEN__ --random-output-len __OUT_LEN__ --random-range-ratio 0.9"
    ["fixed-len"]="--dataset-name fixed-len --fixed-input-len __INPUT_LEN__ --fixed-output-len __OUT_LEN__"
)

# OFFLOAD_RELATED CONFIGURATIONS
OFFLOAD_TYPE=$1 # 0 no offloading, 1: vllm naive offloading, 2: smart_offload
OFFLOAD_DYNAMIC=0 # 0: static offloading, 1: dynamic offloading
OFFLOAD_LAYER_INTER=$2 # used for smart_offload
OFFLOAG_GB=$3 # used for vllm naive offloading
declare -A OFFLOAD_CONFIG=(
    # Format: "OFFLOAD_TYPE "
    ["1"]="--cpu-offload-method default --cpu-offload-gb ${OFFLOAG_GB}"
    # ["2"]="--cpu-offload-method smart_offload --smart-offload-dynamic ${OFFLOAD_DYNAMIC} --smart-offload-interval ${OFFLOAD_LAYER_INTER} --smart-offload-param-target all"
    ["2"]="--cpu-offload-method smart_offload --smart-offload-interval ${OFFLOAD_LAYER_INTER}"
)

# MAX_NUM_BATCHED_TOKENS=(4096)
MAX_NUM_BATCHED_TOKENS=(32768 16384 8192 4096 2048 1024 512) # max number of tokens in a batch, 512, 1024, 2048, 4096, 8192, 16384, 32768
# MAX_NUM_BATCHED_TOKENS=(65536 32768 16384 8192 4096 2048 1024 512) # max number of tokens in a batch, 512, 1024, 2048, 4096, 8192, 16384, 32768
IS_USE_V1=$([ -z "$VLLM_USE_V1" ] || [ "$VLLM_USE_V1" = "0" ] && echo 0 || echo 1)
MAX_NUM_BATCHED_REQS=512

#### Other configurations
SERVE_TYPE=$4 # "online" or "offline"
# EXECUTOR_BACKEND="ray" # "ray" or "mp", for "mp", it only supports on a single node (PP * TP <= 4)
EXECUTOR_BACKEND=$5 # for "mp", it only supports on a single node (PP * TP <= 4) for offline benchmark
PREEMP_MODE="recompute" # For v0, it can be recompute or swap. For v1, it is "recompute"
EN_CHUNKED_PREFILL=True # "True" or "False"
EN_PREFIX_CACHING=False # "True" or "False"
SCHEDULER_CLS=$([ "$IS_USE_V1" == 0 ] && echo "vllm.core.scheduler.Scheduler" || echo "vllm.v1.core.sched.scheduler.Scheduler")

MONITOR_GPU="False" # "True" or "False"

MODEL_NAME=$(echo $MODEL | cut -d'/' -f2)
####################################Get_base_log_path#####################################
get_log_path() {
    if [ $OFFLOAD_TYPE -eq 2 ]; then
        LOG_BASE_PATH="${LOG_BASE_PATH}/smart_offload_v0_8_4/logs_v${IS_USE_V1}_${MODEL_NAME}_ofd${OFFLOAD_TYPE}_int${OFFLOAD_LAYER_INTER}/"
    elif [ $OFFLOAD_TYPE -eq 1 ]; then
        LOG_BASE_PATH="${LOG_BASE_PATH}/venilla_offload_v0_8_4/logs_v${IS_USE_V1}_${MODEL_NAME}_ofd${OFFLOAD_TYPE}_cpu${OFFLOAG_GB}/"
    else
        LOG_BASE_PATH="${LOG_BASE_PATH}/venilla_baseline_v0_8_4/${VLLM_V1_SCHEDULE_METHOD}/logs_v${IS_USE_V1}_${MODEL_NAME}_ofd${OFFLOAD_TYPE}/"
    fi
    SUB_PATH=$([ ${DATASET_NAME} = "longbench" ] && echo "${DATASET_NAME}--${SUBTASK}" || echo "${DATASET_NAME}")

    echo "${LOG_BASE_PATH}/${SUB_PATH}_${SERVE_TYPE}/"
}

get_log_sufix() {
    if [ $OFFLOAD_TYPE -eq 2 ]; then
        # subfix=$([ $OFFLOAD_DYNAMIC -eq 1 ] && echo "dynamic" || echo "static")
        subfix="static"
    elif [ $OFFLOAD_TYPE -eq 1 ]; then
        subfix="cpu${OFFLOAG_GB}"
    else
        subfix=""
    fi

    echo "${subfix}"
}

LOG_PATH=`get_log_path`
[ -d $LOG_PATH ] || mkdir -p $LOG_PATH

###################################### Related Helper functions #############################################
start_gpu_monitor() {
    NRANKS_PER_NODE=1
    echo "Monitoring starting......"
    # start gpu monitor & host memory monitor
    for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); 
    do
        setsid python ${PROJ_PATH}/monitor_gpu.py $gpu_id "${LOG_PATH}/monitor-gpu${gpu_id}.csv" &
        monitor_pid[$gpu_id]=$!
        echo "Monitoring started for GPU $gpu_id at PID ${monitor_pid[$gpu_id]}, PATH=${LOG_PATH}/monitor-gpu${gpu_id}.csv."
    done
    setsid python ${PROJ_PATH}/monitor_host_mem.py "${LOG_PATH}/monitor-vmem.csv" &
}

stop_gpu_monitor() {
    NRANKS_PER_NODE=1
    echo "Monitoring stopping......"
    # Terminate monitoring for all GPUs and host memory
    for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); 
    do
        echo "Killing the monitoring script for GPU $gpu_id at PID ${monitor_pid[$gpu_id]}."
        kill -2 ${monitor_pid[$gpu_id]}
        echo "SIGTERM (kill -2) instructed to monitoring script for GPU $gpu_id."
        wait ${monitor_pid[$gpu_id]}
        echo "Killed the monitoring script for GPU $gpu_id."
    done

    kill -2 $(pgrep -f monitor_host_mem.py)
    echo "SIGTERM (kill -2) instructed to monitoring script for host memory."
    wait $(pgrep -f monitor_host_mem.py)
    echo "Killed the monitoring script for host memory."
}

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
    gpu_mem_limit=$1
    max_num_batched_tokens=$2
    server_log_file=$3
    profile_log_file="${server_log_file}.nsys-rep"
    
    profile_cmd="nsys profile --force-overwrite true -t cuda,cudnn,cublas,nvtx -o ${profile_log_file} --trace-fork-before-exec=true "
    vllm_serve_cmd="vllm serve ${MODEL} \
        --download-dir ${MODEL_PATH} \
        --trust-remote-code \
        --enforce-eager \
        --distributed-executor-backend ${EXECUTOR_BACKEND} \
        --tensor-parallel-size ${TP} \
        --pipeline-parallel-size ${PP} \
        --disable-log-requests \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${gpu_mem_limit} \
        --scheduler-cls ${SCHEDULER_CLS} "
        
    if [ "$IS_USE_V1" == "0" ]; then
        vllm_serve_cmd+=" --preemption-mode ${PREEMP_MODE} --enable-chunked-prefill=${EN_CHUNKED_PREFILL} "
        vllm_serve_cmd+=$([ "$EN_CHUNKED_PREFILL" = "True" ] && echo "--max-num-batched-tokens ${max_num_batched_tokens} " || echo "" ) 
    else
        # vLLM's V1 version
        vllm_serve_cmd+=" --max-num-batched-tokens ${max_num_batched_tokens} " 
    fi

    [ "$EN_PREFIX_CACHING" = "True" ] && vllm_serve_cmd+=" --enable-prefix-caching "
    [ "$EN_PREFIX_CACHING" = "False" ] && vllm_serve_cmd+=" --no-enable-prefix-caching "

    if [ $OFFLOAD_TYPE -ne 0 ]; then
        vllm_serve_cmd+="${OFFLOAD_CONFIG[$OFFLOAD_TYPE]}"
    fi

    # vllm_serve_cmd+=" --collect-layer-fwd-time " 

    # running with nvidia profiler
    # vllm_cmd="${profile_cmd} ${vllm_serve_cmd}"
    # running without nvidia profiler
    vllm_cmd="${vllm_serve_cmd}"
    
    echo "Server command: ${vllm_cmd}"

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
        if grep -q "INFO:     Waiting for application startup." "$server_log_file"; then
            echo "vLLM server started successfully!"
            return 0
        fi
        sleep $sleep_inter
        waiting_time=$((waiting_time + sleep_inter))
    done
}

stop_vllm_server() {
    PIDS=$(pgrep -f "vllm serve")
    if [ -n "$PIDS" ]; then
        echo "Stopping vLLM processes: $PIDS"
        kill -15 $PIDS
        sleep 5
        # Check if any are still running and force kill if necessary
        REMAINING=$(pgrep -f "vllm serve")
        if [ -n "$REMAINING" ]; then
            echo "Force killing remaining vLLM processes: $REMAINING"
            kill -9 $REMAINING
        fi
        echo "VLLM server stopped ..."
    else
        echo "No vLLM processes found"
    fi
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
                "sharegpt")   dataset_config+=" --sharegpt-output-len $gen_len" ;;
            esac
        fi
    fi
    
    client_cmd="python ${EXEC_PATH}/benchmark_serving_v2.py \
                    --backend vllm \
                    --host localhost \
                    --model $MODEL \
                    --ignore-eos \
                    --num-prompts $num_req \
                    ${dataset_config} "

    echo "Client command: ${client_cmd}"

    eval "${client_cmd}" > "${client_log_file_name}" 2>&1
    # sleep 180 # give the server some time to finish the requests
    sleep 60
}

run_offline_bench() {
    local dataset_type=$1
    local client_log_file_name=$2
    local num_req=$3
    local extra_params=$4
    local gpu_mem_limit=$5
    local max_num_batched_tokens=$6

    if [ "$dataset_type" = "synthetic" ]; then
        dataset_config="${extra_params}"
    elif [ "$dataset_type" = "real" ]; then
        dataset_config="${DATASETS[$DATASET_NAME]}" 
        gen_len=$extra_params 
        if [ $gen_len -gt 0 ]; then
            case "$DATASET_NAME" in
                "longbench")   dataset_config+=" --longbench-output-len $gen_len" ;;
                "gsm8k")       dataset_config+=" --gsm8k-output-len $gen_len" ;;
                "sharegpt")   dataset_config+=" --output-len $gen_len" ;;
            esac
        fi
    fi

    client_cmd="python ${EXEC_PATH}/benchmark_throughput_v2.py --backend vllm \
                    --model ${MODEL} --download-dir ${MODEL_PATH} --trust-remote-code \
                    --enforce-eager --distributed-executor-backend ${EXECUTOR_BACKEND} \
                    --tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} \
                    --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${gpu_mem_limit} \
                    --scheduler-cls ${SCHEDULER_CLS} --num-prompts $num_req --disable-log-requests \
                    --max-num-seqs ${MAX_NUM_BATCHED_REQS} --async-engine "  

    if [ "$IS_USE_V1" == "0" ]; then
        client_cmd+=" --preemption-mode ${PREEMP_MODE} --enable-chunked-prefill=${EN_CHUNKED_PREFILL} "
        client_cmd+=$([ "$EN_CHUNKED_PREFILL" = "True" ] && echo "--max-num-batched-tokens ${max_num_batched_tokens} " || echo "" )
    else
        # vLLM's V1 version
        client_cmd+=" --enable-chunked-prefill --max-num-batched-tokens ${max_num_batched_tokens} "
    fi

    [ "$EN_PREFIX_CACHING" = "True" ] && client_cmd+=" --enable-prefix-caching "
    [ "$EN_PREFIX_CACHING" = "False" ] && client_cmd+=" --no-enable-prefix-caching "
 
    if [ $OFFLOAD_TYPE -ne 0 ]; then
        client_cmd+="${OFFLOAD_CONFIG[$OFFLOAD_TYPE]}"
    fi

    client_cmd+=" ${dataset_config} "

    echo "Client command: ${client_cmd}"

    eval "${client_cmd}" > "${client_log_file_name}" 2>&1
    sleep 60
}

################################################## Run the Online Client Test ##########################################################
benchmark_with_real_dataset() {
    local gen_len=$1
    log_sufix=$(get_log_sufix)
    
    for gpu_mem_limit in ${gpu_mem_utils[@]}; do
        GPU_MEM_LIMIT=$(echo "$gpu_mem_limit * 100" | bc)
        for max_num_batched_tokens in ${MAX_NUM_BATCHED_TOKENS[@]}; do
            for num_req in ${NUM_REQS[@]}; do
                echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME}, GPU_MEM_LIMIT=${GPU_MEM_LIMIT}, max_num_batched_tokens=${max_num_batched_tokens}  ..."
                for try_idx in $(seq $S_IDX $NUM_TRIES); do
                    if [ $gen_len -gt 0 ]; then
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/server_c${MAX_MODEL_LEN}_g${gen_len}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                        CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_c${MAX_MODEL_LEN}_g${gen_len}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                    else
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/server_c${MAX_MODEL_LEN}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                        CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_c${MAX_MODEL_LEN}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                    fi

                    # Stop the server first
                    stop_vllm_server

                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        # stop ray cluster in case it is not stopped by previous run
                        stop_ray_cluster
                        # start ray cluster
                        start_ray_cluster
                    fi

                    if [ ${MONITOR_GPU} = "True" ];then
                        start_gpu_monitor
                    fi

                    # Start vLLM server
                    start_vllm_server ${gpu_mem_limit} ${max_num_batched_tokens} ${SERVER_LOG_FILE_NAME} ${try_idx}
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

                    # Stop Monitor
                    if [ ${MONITOR_GPU} = "True" ];then
                        stop_gpu_monitor
                    fi

                    # Stop the vLLM server
                    stop_vllm_server

                    # Stop the ray cluster if it was started
                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        stop_ray_cluster
                    fi
                done
                echo "Finished inference with num_req=${num_req} requests using dataset ${DATASET_NAME} ..."
            done
        done
    done 
}

benchmark_with_synthetic_dataset() {
    log_sufix=$(get_log_sufix)

    local output_len=$1
    if [ "$output_len" = "1" ]; then
        # referring to the prompt-only test case
        OUT_LENS=($(printf "1 %.0s" "${INPUT_LENS[@]}"))
    fi

    for gpu_mem_limit in ${gpu_mem_utils[@]}; do
        GPU_MEM_LIMIT=$(echo "$gpu_mem_limit * 100" | bc)
        for max_num_batched_tokens in ${MAX_NUM_BATCHED_TOKENS[@]}; do
            for num_req in ${NUM_REQS[@]}; do
                echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME}, GPU_MEM_LIMIT=${GPU_MEM_LIMIT}, max_num_batched_tokens=${max_num_batched_tokens} ..."
                echo "aaaa=${!INPUT_LENS[@]}"
                for i in "${!INPUT_LENS[@]}"; do
                    # ensure the input and output lengths are the same
                    input_len=${INPUT_LENS[$i]}
                    out_len=${OUT_LENS[$i]}
                    
                    # Replace placeholders with actual values
                    dataset_config="${SYNT_DATASETS[$DATASET_NAME]}"
                    dataset_config="${dataset_config//__INPUT_LEN__/$input_len}"
                    dataset_config="${dataset_config//__OUT_LEN__/$out_len}"
                    echo "Dataset: $DATASET_NAME, Config: $dataset_config"

                    for try_idx in $(seq $S_IDX $NUM_TRIES); do
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/server_c${MAX_MODEL_LEN}_p${input_len}_g${out_len}_r${num_req}_tp${TP}_pp${PP}_${PREEMP_MODE}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                        CLIENT_LOG_FILE_NAME="${LOG_PATH}/client_c${MAX_MODEL_LEN}_p${input_len}_g${out_len}_r${num_req}_tp${TP}_pp${PP}_${PREEMP_MODE}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"

                        # Stop the server first
                        stop_vllm_server
                        
                        if [ $EXECUTOR_BACKEND = "ray" ]; then
                            # stop ray cluster in case it is not stopped by previous run
                            stop_ray_cluster
                            # start ray cluster
                            start_ray_cluster
                        fi

                        # Start vLLM server
                        start_vllm_server ${gpu_mem_limit} ${max_num_batched_tokens} ${SERVER_LOG_FILE_NAME}
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
    done
}

################################################## Run the Offline Client Test ##########################################################
benchmark_offline_with_real_dataset() {
    local gen_len=$1
    log_sufix=$(get_log_sufix)
    
    for gpu_mem_limit in ${gpu_mem_utils[@]}; do
        GPU_MEM_LIMIT=$(echo "$gpu_mem_limit * 100" | bc)
        for max_num_batched_tokens in ${MAX_NUM_BATCHED_TOKENS[@]}; do
            for num_req in ${NUM_REQS[@]}; do
                echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME}, GPU_MEM_LIMIT=${GPU_MEM_LIMIT}, max_num_batched_tokens=${max_num_batched_tokens}  ..."
                for try_idx in $(seq $S_IDX $NUM_TRIES); do 
                    if [ $gen_len -gt 0 ]; then
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/offline_c${MAX_MODEL_LEN}_g${gen_len}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                    else
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/offline_c${MAX_MODEL_LEN}_r${num_req}_tp${TP}_pp${PP}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"
                    fi

                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        # stop ray cluster in case it is not stopped by previous run
                        stop_ray_cluster
                        # start ray cluster
                        start_ray_cluster
                    fi

                    run_offline_bench "real" ${SERVER_LOG_FILE_NAME} ${num_req} ${gen_len} ${gpu_mem_limit} ${max_num_batched_tokens} ${try_idx} 
                    sleep 2
                    # Stop the ray cluster if it was started
                    if [ $EXECUTOR_BACKEND = "ray" ]; then
                        stop_ray_cluster
                    fi
                done
                echo "Finished inference with num_req=${num_req} requests using dataset ${DATASET_NAME} ..."
            done
        done
    done 
}

benchmark_offline_with_synthetic_dataset() {
    log_sufix=$(get_log_sufix)

    local output_len=$1
    if [ "$output_len" = "1" ]; then
        # referring to the prompt-only test case
        OUT_LENS=($(printf "1 %.0s" "${INPUT_LENS[@]}"))
    fi

    for gpu_mem_limit in ${gpu_mem_utils[@]}; do
        GPU_MEM_LIMIT=$(echo "$gpu_mem_limit * 100" | bc)
        for max_num_batched_tokens in ${MAX_NUM_BATCHED_TOKENS[@]}; do
            for num_req in ${NUM_REQS[@]}; do
                echo "Start running with num_req=${num_req} requests using dataset ${DATASET_NAME}, GPU_MEM_LIMIT=${GPU_MEM_LIMIT}, max_num_batched_tokens=${max_num_batched_tokens} ..."
                echo "aaaa=${!INPUT_LENS[@]}"
                for i in "${!INPUT_LENS[@]}"; do
                    # ensure the input and output lengths are the same
                    input_len=${INPUT_LENS[$i]}
                    out_len=${OUT_LENS[$i]}
                    
                    # Replace placeholders with actual values
                    dataset_config="${SYNT_DATASETS[$DATASET_NAME]}"
                    dataset_config="${dataset_config//__INPUT_LEN__/$input_len}"
                    dataset_config="${dataset_config//__OUT_LEN__/$out_len}"
                    echo "Dataset: $DATASET_NAME, Config: $dataset_config"

                    for try_idx in $(seq $S_IDX $NUM_TRIES); do
                        SERVER_LOG_FILE_NAME="${LOG_PATH}/offline_c${MAX_MODEL_LEN}_p${input_len}_g${out_len}_r${num_req}_tp${TP}_pp${PP}_${PREEMP_MODE}_gpu${gpu_mem_limit}_bt${max_num_batched_tokens}_${try_idx}_${log_sufix}.log"

                        if [ $EXECUTOR_BACKEND = "ray" ]; then
                            # stop ray cluster in case it is not stopped by previous run
                            stop_ray_cluster
                            # start ray cluster
                            start_ray_cluster
                        fi
                        
                        run_offline_bench "synthetic" ${SERVER_LOG_FILE_NAME} ${num_req} "${dataset_config}" ${gpu_mem_limit} ${max_num_batched_tokens} ${try_idx} 
                        
                        sleep 2

                        # Stop the ray cluster if it was started
                        if [ $EXECUTOR_BACKEND = "ray" ]; then
                            stop_ray_cluster
                        fi
                    done
                done
            done
        done
    done
}



#################################################################################
for test_case in "${TESTA_CASES[@]}"; do
    if [ "$test_case" = "prompt-only" ]; then
        gen_len=1 # decode length
    elif [ "$test_case" = "prompt-decode" ]; then
        if [ "$DATASET_NAME" = "longbench" ]; then
            gen_len=$(jq '.'${SUBTASK} $EXEC_PATH/longbench/config/dataset2maxlen.json)
        else
            gen_len=-1 # decode length, meaning that we did not override the decode length
        fi
    else
        echo "Invalid test case ${test_case}. Continue next..."
        continue
    fi
    echo "gen_len=$gen_len"
    case "$DATASET_NAME" in
        "longbench"|"gsm8k"|"sharegpt")   
            if [ $SERVE_TYPE = "online" ]; then
                benchmark_with_real_dataset $gen_len 
            else
                # offline benchmark
                benchmark_offline_with_real_dataset $gen_len 
            fi 
            ;;
        "random"|"fixed-len")
            if [ $SERVE_TYPE = "online" ]; then
                benchmark_with_synthetic_dataset $gen_len 
            else
                # offline benchmark
                benchmark_with_synthetic_dataset $gen_len 
            fi 
            ;;
        *)
            echo "Unknown dataset: $DATASET_NAME" >&2
            exit 1
        ;;
    esac  
done 
