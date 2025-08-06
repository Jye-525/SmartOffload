import re, sys, csv, os
from typing import List, Dict, Tuple, Union
from collections import defaultdict
from datetime import datetime
from file_read_backwards import FileReadBackwards
 
###### Get the scheduler info of each step from the log file ######
def Extract_schedule_info(line: str, scheduler_info: Dict[int, Dict[str, Union[int, float]]]):
    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?EngineCore step\(\) scheduled\s+(\d+).*?new requests.*?\s+(\d+).*?cached requests.*?step_id\s+(\d+)\s+.*?total scheduled tokens\s+(\d+).*?KV cache utilization\s+(\d+\.\d+)%.*?', line.strip())
    if match:
        # Extract the number of recompute tokens from the log line
        datetime_str = match.group(1) 
        step_id = int(match.group(4))
        new_reqs = int(match.group(2))
        cached_reqs = int(match.group(3))
        num_scheduled_tokens = int(match.group(5))
        kv_cache_utilization = float(match.group(6))
        if num_scheduled_tokens > 0:
            dt = datetime.strptime(f'2025-{datetime_str}', '%Y-%m-%d %H:%M:%S')  
            timestamp = dt.timestamp()
            scheduler_info[step_id] = {
                'step_id': step_id,
                'timestamp': timestamp,
                'timestamp_str': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'new_reqs': new_reqs,
                'cached_reqs': cached_reqs,
                'num_scheduled_tokens': num_scheduled_tokens,
                'kv_cache_utilization': kv_cache_utilization
            }
        else:
            # If the number of scheduled tokens is 0, we skip this step
            print(f"Step {step_id} has no scheduled tokens, skipping...")
    else:
        # Handle the case where the log line format is different
        print(f"Error: Unexpected log line format: {line.strip()}")
        
def Extract_resumed_info(line: str, resumed_info: Dict[int, Dict[str, Union[int, float]]]):
    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?Scheduler scheduled\s+(\d+)\s+resumed requests with\s+(\d+)\s+tokens on step\s+(\d+)', line.strip())
    if match:
        # Extract the number of recompute tokens from the log line
        datetime_str = match.group(1) 
        resumed_reqs = int(match.group(2))
        recmp_tokens = int(match.group(3))
        step_id = int(match.group(4))
        
        dt = datetime.strptime(f'2025-{datetime_str}', '%Y-%m-%d %H:%M:%S')  
        resumed_info[step_id] = {
            'step_id': step_id,
            'resumed_reqs': resumed_reqs,
            'recmp_tokens': recmp_tokens,
        }
    else:
        # Handle the case where the log line format is different
        print(f"Error: Unexpected log line format: {line.strip()}")    

def Extract_preempted_info(line: str, preempt_info: Dict[int, Dict[str, Union[int, float]]]):
    # INFO 06-13 04:25:57 [scheduler.py:556] Scheduler preempted 1 requests with 949 tokens on step 3146
    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?Scheduler preempted\s+(\d+)\s+requests with\s+(\d+)\s+tokens on step\s+(\d+)', line.strip())
    if match:
        # Extract the number of recompute tokens from the log line
        datetime_str = match.group(1)
        preempted_reqs = int(match.group(2))
        preempted_tokens = int(match.group(3))
        step_id = int(match.group(4))

        dt = datetime.strptime(f'2025-{datetime_str}', '%Y-%m-%d %H:%M:%S')
        preempt_info[step_id] = {
            'step_id': step_id,
            'preempted_reqs': preempted_reqs,
            'preempted_tokens': preempted_tokens,
        }
        
def Extract_schedule_end_info(line: str, scheduler_end_info: Dict[int, Dict[str, Union[int, float]]]):
    # INFO 07-17 19:31:10 [core.py:222] EngineCore step() finished the forward iteration in step_id 196 Scheduler stats: 322 running requests, 4279 waiting requests, total scheduled tokens 325, KV cache utilization 98.14% sched_time = 0.84 ms exec_time = 44.34 ms sched+exec = 45.17 ms per_step_time = 45.23 ms
    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?EngineCore step\(\) finished the forward iteration in step_id\s+(\d+)\s+Scheduler stats:.*?sched_time\s+=\s+([\d.]+)\s+ms\s+exec_time\s+=\s+([\d.]+)\s+ms\s+sched\+exec\s+=\s+([\d.]+)\s+ms\s+per_step_time\s+=\s+([\d.]+)\s+ms', line.strip())
    if match:
        datetime_str = match.group(1)
        step_id = int(match.group(2))
        sched_time = float(match.group(3))
        exec_time = float(match.group(4))
        sched_exec_time = float(match.group(5))
        per_step_time = float(match.group(6))
        scheduler_end_info[step_id] = {
            'step_id': step_id,
            'sched_time': sched_time,
            'exec_time': exec_time,
            'sched_exec_time': sched_exec_time,
            'per_step_time': per_step_time,
        }
    else:
        # Handle the case where the log line format is different
        print(f"Error: Unexpected log line format: {line.strip()}")
    
    
def Extract_scheduled_reqs_per_iter(log_file, pp_ranks):
    # Parse the corresponding log file and check number of requests scheduled by each scheduler during each iteration step
    # EngineCore step() scheduled 1 new requests, 0 cached requests in the the step function, step_id 1 ,total scheduled tokens 256, KV cache utilization 0.30%
    assert pp_ranks == 1, "The pp_ranks should be 1 for this function"
    scheduler_info = {}
    preempt_info = {}
    resumed_info = {}
    # We need to skip the step_id if the number of scheduled tokens is 0
    with open(log_file, 'r') as file:
        for line in file:
            if 'EngineCore step() scheduled' in line:
                Extract_schedule_info(line, scheduler_info)
            elif 'Scheduler preempted' in line:
                Extract_preempted_info(line, preempt_info)
            elif 'resumed requests with' in line:
                Extract_resumed_info(line, resumed_info)
            else:
                continue
            
    # get the last step_id from the scheuler_info
    if scheduler_info:
        last_step_id = max(scheduler_info.keys())
        print(f"Last step_id found in the log file: {last_step_id}")
        # re-open the log file again to check the last step_id finished time
        with FileReadBackwards(log_file, encoding="utf-8") as frb:
            for line in frb:
                if 'EngineCore step() finished the forward iteration' in line:
                    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?step_id\s+(\d+)', line.strip())
                    if match:
                        step_id = int(match.group(2))
                        if step_id == last_step_id:
                            datetime_str = match.group(1)
                            dt = datetime.strptime(f'2025-{datetime_str}', '%Y-%m-%d %H:%M:%S')
                            timestamp = dt.timestamp()
                            scheduler_info[last_step_id]['timestamp'] = timestamp
                            print(f"Last step_id {last_step_id} finished at {timestamp} seconds")
                            break
                    
    return scheduler_info, preempt_info, resumed_info

def write_schedule_to_csv(schedule_info: Dict[int, Dict[str, Union[int, float]]], 
                          preempt_info: Dict[int, Dict[str, Union[int, float]]],
                          resumed_info: Dict[int, Dict[str, Union[int, float]]],
                          output_csv_file: str):
    """
    Write the schedule information to a CSV file.
    # schedule_info include the schedule infor of each step (including kv utilization)
    # preempt_info include the preempted requests information when preesmp happened
    # resumed_info include the resumed requests information when resumed happened
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['date_time','time(s)', 'step_id', 'new_reqs', 'cached_reqs', 'scheduled_reqs', 'num_scheduled_tokens', 'kv_cache_utilization', 'preempted_reqs', 'preempted_tokens', 'resumed_reqs', 'recmp_tokens']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        id = 1
        first_req = schedule_info.get(1, {})
        start_timestamp = first_req.get('timestamp', 0)
        tmp_reqs = []
        last_relative_time = 0
        for step_id, sched_info in schedule_info.items():
            relative_time = sched_info['timestamp'] - start_timestamp
            if (id == 1) or (relative_time == last_relative_time):
                tmp_reqs.append(sched_info)
                last_relative_time = relative_time
            else:
                # the duration of these request is the timestamp of the first request in this step
                relative_time_1 = tmp_reqs[0]['timestamp'] - start_timestamp
                dur = relative_time - relative_time_1
                inter = dur/len(tmp_reqs)
                acc_offset = 0
                for tmp_info in tmp_reqs:
                    tmp_step_id = tmp_info['step_id']
                    ### compose the row information
                    if tmp_step_id in preempt_info:
                        preempted_reqs = preempt_info[tmp_step_id]['preempted_reqs']
                        preempted_tokens = preempt_info[tmp_step_id]['preempted_tokens']
                    else:
                        preempted_reqs, preempted_tokens = 0, 0
                        
                    if tmp_step_id in resumed_info:
                        resumed_reqs = resumed_info[tmp_step_id]['resumed_reqs']
                        recmp_tokens = resumed_info[tmp_step_id]['recmp_tokens']
                    else:
                        resumed_reqs, recmp_tokens = 0, 0
                    
                    writer.writerow({
                        'time(s)': relative_time_1 + acc_offset,
                        'date_time': tmp_info['timestamp_str'],
                        'step_id': tmp_info['step_id'],
                        'new_reqs': tmp_info['new_reqs'],
                        'cached_reqs': tmp_info['cached_reqs'],
                        'scheduled_reqs': tmp_info['new_reqs'] + tmp_info['cached_reqs'],
                        'num_scheduled_tokens': tmp_info['num_scheduled_tokens'],
                        'kv_cache_utilization': tmp_info['kv_cache_utilization'],
                        'preempted_reqs': preempted_reqs,
                        'preempted_tokens': preempted_tokens,
                        'resumed_reqs': resumed_reqs,
                        'recmp_tokens': recmp_tokens
                    })
                    acc_offset += inter
                
                tmp_reqs.clear()
                tmp_reqs.append(sched_info)
                last_relative_time = relative_time
            id += 1
            
        if tmp_reqs:
            # write the last group of requests
            relative_time_1 = tmp_reqs[0]['timestamp'] - start_timestamp
            relative_time = tmp_reqs[-1]['timestamp'] - start_timestamp
            dur = relative_time - relative_time_1 + (0.10) 
            inter = dur/len(tmp_reqs)
            acc_offset = 0
            preempted_reqs, preempted_tokens = 0, 0
            resumed_reqs, recmp_tokens = 0, 0
            for tmp_info in tmp_reqs:
                tmp_step_id = tmp_info['step_id']
                if tmp_step_id in preempt_info:
                    preempted_reqs = preempt_info[tmp_step_id]['preempted_reqs']
                    preempted_tokens = preempt_info[tmp_step_id]['preempted_tokens']
                else:
                    preempted_reqs, preempted_tokens = 0, 0
                        
                if tmp_step_id in resumed_info:
                    resumed_reqs = resumed_info[tmp_step_id]['resumed_reqs']
                    recmp_tokens = resumed_info[tmp_step_id]['recmp_tokens']
                else:
                    resumed_reqs, recmp_tokens = 0, 0
                
                writer.writerow({
                    'time(s)': relative_time_1 + acc_offset,
                    'date_time': tmp_info['timestamp_str'],
                    'step_id': tmp_info['step_id'],
                    'new_reqs': tmp_info['new_reqs'],
                    'cached_reqs': tmp_info['cached_reqs'],
                    'scheduled_reqs': tmp_info['new_reqs'] + tmp_info['cached_reqs'],
                    'num_scheduled_tokens': tmp_info['num_scheduled_tokens'],
                    'kv_cache_utilization': tmp_info['kv_cache_utilization'],
                    'preempted_reqs': preempted_reqs,
                    'preempted_tokens': preempted_tokens,
                    'resumed_reqs': resumed_reqs,
                    'recmp_tokens': recmp_tokens
                })
                acc_offset += inter
    print(f"Schedule information results written to {output_csv_file}")
    
    
def Extract_scheduled_reqs_per_iter_1(log_file, pp_ranks):
    # Parse the corresponding log file and check number of requests scheduled by each scheduler during each iteration step
    # EngineCore step() scheduled 1 new requests, 0 cached requests in the the step function, step_id 1 ,total scheduled tokens 256, KV cache utilization 0.30%
    assert pp_ranks == 1, "The pp_ranks should be 1 for this function"
    scheduler_info = {}
    preempt_info = {}
    resumed_info = {}
    scheduler_end_info = {}
    # We need to skip the step_id if the number of scheduled tokens is 0
    with open(log_file, 'r') as file:
        for line in file:
            if 'EngineCore step() scheduled' in line:
                Extract_schedule_info(line, scheduler_info)
            elif 'Scheduler preempted' in line:
                Extract_preempted_info(line, preempt_info)
            elif 'resumed requests with' in line:
                Extract_resumed_info(line, resumed_info)
            elif 'EngineCore step() finished the forward iteration' in line:
                Extract_schedule_end_info(line, scheduler_end_info)
            else:
                continue
            
    # get the last step_id from the scheuler_info
    if scheduler_info:
        last_step_id = max(scheduler_info.keys())
        print(f"Last step_id found in the log file: {last_step_id}")
        # re-open the log file again to check the last step_id finished time
        with FileReadBackwards(log_file, encoding="utf-8") as frb:
            for line in frb:
                if 'EngineCore step() finished the forward iteration' in line:
                    match = re.search(r'INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?step_id\s+(\d+)', line.strip())
                    if match:
                        step_id = int(match.group(2))
                        if step_id == last_step_id:
                            datetime_str = match.group(1)
                            dt = datetime.strptime(f'2025-{datetime_str}', '%Y-%m-%d %H:%M:%S')
                            timestamp = dt.timestamp()
                            scheduler_info[last_step_id]['timestamp'] = timestamp
                            print(f"Last step_id {last_step_id} finished at {timestamp} seconds")
                            break

    return scheduler_info, preempt_info, resumed_info, scheduler_end_info

def write_schedule_to_csv_1(schedule_info: Dict[int, Dict[str, Union[int, float]]], 
                          preempt_info: Dict[int, Dict[str, Union[int, float]]],
                          resumed_info: Dict[int, Dict[str, Union[int, float]]],
                          scheduler_end_info: Dict[int, Dict[str, Union[int, float]]],
                          output_csv_file: str):
    """
    Write the schedule information to a CSV file.
    # schedule_info include the schedule infor of each step (including kv utilization)
    # preempt_info include the preempted requests information when preesmp happened
    # resumed_info include the resumed requests information when resumed happened
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['date_time','time(s)', 'step_id', 'new_reqs', 'cached_reqs', 'scheduled_reqs', 'num_scheduled_tokens', 'kv_cache_utilization', 'preempted_reqs', 'preempted_tokens', 'resumed_reqs', 'recmp_tokens', 'sched_time', 'exec_time', 'sched_exec_time', 'per_step_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        id = 1
        first_req = schedule_info.get(1, {})
        start_timestamp = first_req.get('timestamp', 0)
        tmp_reqs = []
        last_relative_time = 0
        for step_id, sched_info in schedule_info.items():
            relative_time = sched_info['timestamp'] - start_timestamp
            if (id == 1) or (relative_time == last_relative_time):
                tmp_reqs.append(sched_info)
                last_relative_time = relative_time
            else:
                # the duration of these request is the timestamp of the first request in this step
                relative_time_1 = tmp_reqs[0]['timestamp'] - start_timestamp
                dur = relative_time - relative_time_1
                inter = dur/len(tmp_reqs)
                acc_offset = 0
                for tmp_info in tmp_reqs:
                    tmp_step_id = tmp_info['step_id']
                    ### compose the row information
                    if tmp_step_id in preempt_info:
                        preempted_reqs = preempt_info[tmp_step_id]['preempted_reqs']
                        preempted_tokens = preempt_info[tmp_step_id]['preempted_tokens']
                    else:
                        preempted_reqs, preempted_tokens = 0, 0
                        
                    if tmp_step_id in resumed_info:
                        resumed_reqs = resumed_info[tmp_step_id]['resumed_reqs']
                        recmp_tokens = resumed_info[tmp_step_id]['recmp_tokens']
                    else:
                        resumed_reqs, recmp_tokens = 0, 0
                        
                    sched_time = scheduler_end_info[tmp_step_id]['sched_time']
                    exec_time = scheduler_end_info[tmp_step_id]['exec_time']
                    sched_exec_time = scheduler_end_info[tmp_step_id]['sched_exec_time']
                    per_step_time = scheduler_end_info[tmp_step_id]['per_step_time']
                    
                    writer.writerow({
                        'time(s)': relative_time_1 + acc_offset,
                        'date_time': tmp_info['timestamp_str'],
                        'step_id': tmp_info['step_id'],
                        'new_reqs': tmp_info['new_reqs'],
                        'cached_reqs': tmp_info['cached_reqs'],
                        'scheduled_reqs': tmp_info['new_reqs'] + tmp_info['cached_reqs'],
                        'num_scheduled_tokens': tmp_info['num_scheduled_tokens'],
                        'kv_cache_utilization': tmp_info['kv_cache_utilization'],
                        'preempted_reqs': preempted_reqs,
                        'preempted_tokens': preempted_tokens,
                        'resumed_reqs': resumed_reqs,
                        'recmp_tokens': recmp_tokens,
                        'sched_time': sched_time,
                        'exec_time': exec_time,
                        'sched_exec_time': sched_exec_time,
                        'per_step_time': per_step_time
                    })
                    acc_offset += inter
                
                tmp_reqs.clear()
                tmp_reqs.append(sched_info)
                last_relative_time = relative_time
            id += 1
            
        if tmp_reqs:
            # write the last group of requests
            relative_time_1 = tmp_reqs[0]['timestamp'] - start_timestamp
            relative_time = tmp_reqs[-1]['timestamp'] - start_timestamp
            dur = relative_time - relative_time_1 + (0.10) 
            inter = dur/len(tmp_reqs)
            acc_offset = 0
            preempted_reqs, preempted_tokens = 0, 0
            resumed_reqs, recmp_tokens = 0, 0
            for tmp_info in tmp_reqs:
                tmp_step_id = tmp_info['step_id']
                if tmp_step_id in preempt_info:
                    preempted_reqs = preempt_info[tmp_step_id]['preempted_reqs']
                    preempted_tokens = preempt_info[tmp_step_id]['preempted_tokens']
                else:
                    preempted_reqs, preempted_tokens = 0, 0
                        
                if tmp_step_id in resumed_info:
                    resumed_reqs = resumed_info[tmp_step_id]['resumed_reqs']
                    recmp_tokens = resumed_info[tmp_step_id]['recmp_tokens']
                else:
                    resumed_reqs, recmp_tokens = 0, 0
                
                sched_time = scheduler_end_info[tmp_step_id]['sched_time']
                exec_time = scheduler_end_info[tmp_step_id]['exec_time']
                sched_exec_time = scheduler_end_info[tmp_step_id]['sched_exec_time']
                per_step_time = scheduler_end_info[tmp_step_id]['per_step_time']
                 
                writer.writerow({
                    'time(s)': relative_time_1 + acc_offset,
                    'date_time': tmp_info['timestamp_str'],
                    'step_id': tmp_info['step_id'],
                    'new_reqs': tmp_info['new_reqs'],
                    'cached_reqs': tmp_info['cached_reqs'],
                    'scheduled_reqs': tmp_info['new_reqs'] + tmp_info['cached_reqs'],
                    'num_scheduled_tokens': tmp_info['num_scheduled_tokens'],
                    'kv_cache_utilization': tmp_info['kv_cache_utilization'],
                    'preempted_reqs': preempted_reqs,
                    'preempted_tokens': preempted_tokens,
                    'resumed_reqs': resumed_reqs,
                    'recmp_tokens': recmp_tokens,
                    'sched_time': sched_time,
                    'exec_time': exec_time,
                    'sched_exec_time': sched_exec_time,
                    'per_step_time': per_step_time
                })
                acc_offset += inter
    print(f"Schedule information results written to {output_csv_file}")

########################Extract each request preempt/resuemd information from the log file########################
def Extract_req_preempt_info(log_file, pp_ranks):
    # Parse the corresponding log file and check number of requests scheduled by each scheduler during each iteration step
    # EngineCore step() scheduled 1 new requests, 0 cached requests in the the step function, step_id 1 ,total scheduled tokens 256, KV cache utilization 0.30%
    assert pp_ranks == 1, "The pp_ranks should be 1 for this function"
    preempt_reqs_info = {}
    # We need to skip the step_id if the number of scheduled tokens is 0
    with open(log_file, 'r') as file:
        for line in file:
            if 'Repeated preempt count:' in line:
                ### INFO 06-30 15:48:34 [request.py:162] Request test689 is finished. Raw prompt token len: 7, Repeated preempt count: 4, Preempt step IDs: [178, 202, 217, 225], Resume step IDs: [198, 210, 223, 226]
                # Extract the scheduler ID from the log line
                match = re.search(
                    r"Request (?P<request_id>\w+) is finished\. "
                    r"Raw prompt token len: (?P<prompt_token_len>\d+), "
                    r"Repeated preempt count: (?P<preempt_count>\d+), "
                    r"Preempt step IDs: \[(?P<preempt_ids>[^\]]+)\], "
                    r"Resume step IDs: \[(?P<resume_ids>[^\]]+)\]", 
                    line.strip()
                )
                
                if match:
                    # Extract the number of recompute tokens from the log line
                    request_id = match.group("request_id")
                    prompt_token_len = int(match.group("prompt_token_len"))
                    preempt_count = int(match.group("preempt_count"))

                    # extract the preempt step IDs and resume step IDs
                    preempt_ids = [int(x.strip()) for x in match.group("preempt_ids").split(',')]
                    resume_ids = [int(x.strip()) for x in match.group("resume_ids").split(',')]
                    
                    preempt_reqs_info[request_id] = {
                        'req_id': request_id,
                        'num_prompts': prompt_token_len,
                        'preempt_count': preempt_count,
                        'preempt_step_ids': preempt_ids,
                        'resume_step_ids': resume_ids
                    }
                else:
                    # Handle the case where the log line format is different
                    print(f"Error: Unexpected log line format: {line.strip()}")
                    continue 

    return preempt_reqs_info

def write_preempt_reqs_to_csv(preempt_reqs_info: Dict[int, Dict[str, Union[int, float]]], output_csv_file: str):
    """
    Write the schedule of preempted request information to a CSV file.
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['req_id', 'num_prompts', 'preempt_count', 'preempt_step_ids', 'resume_step_ids']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        id = 1
        for req_id, info in preempt_reqs_info.items():
            writer.writerow({
                'req_id': info['req_id'],
                'num_prompts': info['num_prompts'],
                'preempt_count': info['preempt_count'],
                'preempt_step_ids': info['preempt_step_ids'],
                'resume_step_ids': info['resume_step_ids']
            })
            id += 1
            
    print(f"Schedule information results written to {output_csv_file}")


if __name__ == "__main__":
    # get the parameters from the command line
    if len(sys.argv) != 6:
        print("Usage: python parse_log_vllm_v1.py <logfile> <model_name> <dataset> <output_dir> <eviction_method>")
        sys.exit(1)

    logfile = sys.argv[1]
    model_name = sys.argv[2]
    dataset = sys.argv[3]
    output_dir = sys.argv[4]
    eviction_method = sys.argv[5]

    # Extract model name, PP and TP rank, prompt_length, gen_length, batch size, gpu_mem from the log file
    base_dir = os.path.dirname(logfile)
    file_name=os.path.basename(logfile)
    # Split into parts
    parts = file_name.split('_')

    # Extract fields, skip parts[10]=eager
    context_len = int(parts[1].replace('c', ''))  # 2048 (from "c2048")
    if parts[2].startswith('g'):
        gen_len = int(parts[2].replace('g', ''))  # 1 (from "g1")
        requests = int(parts[3].replace('r', ''))  # 1 (from "r1")
        tp_ranks = int(parts[4].replace('tp', ''))  # 2 (from "tp2")
        pp_ranks = int(parts[5].replace('pp', ''))  # 2 (from "pp2")
        gpu = float(parts[6].replace('gpu', ''))  # 0.8 (from "gpu0.8")
        bt = int(parts[7].replace('bt', ''))  # 1 (from "bt1")
        try_idx = int(parts[8])
    else:
        # dynamic output length
        gen_len = -1
        requests = int(parts[2].replace('r', ''))  # 1 (from "r1")
        tp_ranks = int(parts[3].replace('tp', ''))  # 2 (from "tp2")
        pp_ranks = int(parts[4].replace('pp', ''))  # 2 (from "pp2")
        gpu = float(parts[5].replace('gpu', ''))  # 0.8 (from "gpu0.8")
        bt = int(parts[6].replace('bt', ''))  # 1 (from "bt1")
        try_idx = int(parts[7])  # 0 (from "0")

    print({
        "model_name": model_name,
        "gen_len": gen_len,
        "requests": requests,
        "tp_ranks": tp_ranks,
        "pp_ranks": pp_ranks,
        "gpu": gpu,
        "bt": bt,
        "try_idx": try_idx
    }) 
    
    # get the base_csv_file_name
    # base_csv_dir= os.path.join(base_dir, f"csv_results")
    base_csv_dir = os.path.join(output_dir, f"csv_results/{eviction_method}")
    os.makedirs(base_csv_dir, exist_ok=True)
    if gen_len == -1:
        # Dynamic output length
        base_csv_file_prefix = f"{model_name}_{dataset}_c{context_len}_r{requests}_tp{tp_ranks}_pp{pp_ranks}_gpu{gpu}_bt{bt}_{try_idx}"
    else:
        base_csv_file_prefix= f"{model_name}_{dataset}_g{gen_len}_r{requests}_tp{tp_ranks}_pp{pp_ranks}_gpu{gpu}_bt{bt}_{try_idx}"
    
    # Extract the scheduler information of each step
    # schedule_per_step = Extract_scheduled_reqs_per_iter(logfile, 1)
    # # Write the schedule info to a CSV file
    # schedule_csv_file = os.path.join(output_dir, f"{base_csv_file_prefix}_sche.csv") 
    # write_schedule_to_csv(schedule_per_step, schedule_csv_file)
    
    # # Extract the scheduler information of each step
    # resumed_info_per_step = Extract_resumed_reqs_per_iter(logfile, 1)
    # # Write the schedule info to a CSV file
    # resumed_csv_file = os.path.join(output_dir, f"{base_csv_file_prefix}_recmp.csv") 
    # write_resumed_reqs_to_csv(resumed_info_per_step, resumed_csv_file) 
    
    # Extract the scheduler information of each step
    # schedule_per_step, preempt_info, resumed_info = Extract_scheduled_reqs_per_iter(logfile, pp_ranks)
    # write_schedule_to_csv(schedule_per_step, preempt_info, resumed_info,
    #                       os.path.join(output_dir, f"{base_csv_file_prefix}_merged.csv"))
    
    
    schedule_per_step, preempt_info, resumed_info, schedule_end_info = Extract_scheduled_reqs_per_iter_1(logfile, pp_ranks)
    write_schedule_to_csv_1(schedule_per_step, preempt_info, resumed_info, schedule_end_info,
                          os.path.join(base_csv_dir, f"{base_csv_file_prefix}_merged.csv"))
