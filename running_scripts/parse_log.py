import re, sys, csv

#logfile = "/home/jieye/moe_mix_precision/SmartOffload_polaris/running_scripts/logs_perf_model/server_deepseek-coder-33b-base_fixed_p511_g1_r1_tp1_pp4_eager.log"
logfile = sys.argv[1]
parse_log = sys.argv[2]
assert parse_log in ["prompt", "decode"], "Invalid log type. Please choose from 'prompt' or 'decode'."
print(f"Parsing log file: {logfile}")
recv_fwd_list = sys.argv[3]
print(f"Forward list: {recv_fwd_list}")
fwd_list = numbers = [int(num) for num in recv_fwd_list.split()]
prompt_len=int(sys.argv[4])


PP_Ranks = 4
target_results = {}
captured_results = {}

# parse the prompt stage infomation
def parse_and_extract_target_lines(file_name, fwd_id, parse_type="prompt"):
    post_count = 0
    capture = False
    # Define regex pattern to search for "start forward..." and "stop forward..."
    pattern_start_fwd = r"start forward on the model, fwd_counts: (\d+) input_ids\.shape: (\d+)"
    pattern_stop_fwd = r"stop forward on the model, fwd_counts: (\d+) input_ids\.shape: (\d+)"
    # Define regex pattern to extract data
    target_pattern = r"PP Rank (\d+).*?elapsed time: (\d+\.\d+) ms.*?start_layer: (\d+).*?end_layer: (\d+).*?fwd_counts: (\d+).*?input_ids\.shape: (\d+)"

    if parse_type == "decode":
        post_count_limit = 1
    else:
        post_count_limit = 2
    
    # Start search from the second "POST /v1"
    with open(file_name, "r") as f:
        for line in f:
            if "POST /v1" in line:
                post_count += 1
            if post_count < post_count_limit:
                continue
            # After 2nd POST /v1, wait for "start forward..."
            match_start_fwd = re.search(pattern_start_fwd, line)
            if match_start_fwd:
                fwd_counts = int(match_start_fwd.group(1))
                if fwd_counts == fwd_id:
                    capture = True
                    continue
            
            if capture:
                # find the stop capture point
                match_stop_fwd = re.search(pattern_stop_fwd, line)
                if match_stop_fwd:
                    fwd_counts = int(match_stop_fwd.group(1))
                    if fwd_counts == fwd_id:
                        break
                    
                # check if the line match the following pattern and fwd_counts match fwd_id, if yes, put this line to the output list
                match = re.search(target_pattern, line.strip())
                if match:
                    # get the fwd_counts
                    fwd_counts = int(match.group(5))
                    if fwd_counts == fwd_id:
                        captured_results.setdefault(fwd_id, []).append(line.strip())


def parse_target_fwd_info(fwd_id, prompt_len, target_lines):
    # Define regex pattern to extract data
    target_pattern = r"PP Rank (\d+).*?elapsed time: (\d+\.\d+) ms.*?start_layer: (\d+).*?end_layer: (\d+).*?fwd_counts: (\d+).*?input_ids\.shape: (\d+)"
    
    target_results = {} 
    for per_layer_line in target_lines:
        match = re.search(target_pattern, per_layer_line)
        if match:
            rank_id = int(match.group(1))          # "0"
            elapsed_time = float(match.group(2))     # "3.848"
            start_layer = int(match.group(3))      # "0"
            end_layer = int(match.group(4))        # "15"
            fwd_counts = int(match.group(5))
            total_tokens = int(match.group(6))
            
            assert fwd_counts == fwd_id, f"Error: fwd_counts {fwd_counts} does not match fwd_id {fwd_id}"
            layers = end_layer - start_layer
            num_reqs = total_tokens // prompt_len
            
            if rank_id in target_results:
                target_results[rank_id]['total_time'] += elapsed_time
            else:
                taget_result = {
                    'layers': layers, 
                    'num_reqs': num_reqs,
                    'total_time': elapsed_time,
                }
                target_results[rank_id] = taget_result
        else:
            print("WARN: No match found.")
    
    # calculate the avergate time per rank
    for rank_id in target_results:
        target_results[rank_id]['avg_time'] = target_results[rank_id]["total_time"] / target_results[rank_id]["layers"]
            
    return target_results

def cal_avg_time(per_fwd_result, PP_Ranks):
    assert len(per_fwd_result) == PP_Ranks, f"Error: The length of per_fwd_result {len(per_fwd_result)} does not match PP_Ranks {PP_Ranks}"
    num_reqs = per_fwd_result[0]['num_reqs']
    total_layers = 0
    global_avg_time = 0
    global_total_time = 0
    for i in range(PP_Ranks):
        total_layers += per_fwd_result[i]["layers"]
        global_total_time += per_fwd_result[i]["total_time"]

    global_avg_time = global_total_time / total_layers
    return num_reqs, total_layers, global_total_time, global_avg_time

def write_to_csv(csv_file, target_results):
    csv_header = ["fwd_id", "rank_id", "layers", "num_reqs", "total_time", "avg_time"]
    rows = []
    # papre the rows
    for item in target_results:
        for rank_id, metrics in item['result'].items():
            rows.append([
                item['fwd_id'],          # fwd_id
                rank_id,                 # rank_id
                metrics['layers'],       # layers
                metrics['num_reqs'],     # num_reqs
                metrics['total_time'],   # total_time
                metrics['avg_time']     # avg_time
            ])
    
        # Write to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)  # Write header
        writer.writerows(rows)       # Write all data rows

# def parse_decode_stage_info(file_name, skip_model_fwd_counts=5):
#     post_count = 0
#     model_fwd_counts = 0
#     capture = False
#     capture_counts = 0
#     # Define regex pattern to extract data
#     pattern = r"PP Rank (\d+).*?elapsed time: (\d+\.\d+) ms.*?start_layer: (\d+).*?end_layer: (\d+)"

#     # Start search from the second "POST /v1"
#     with open(file_name, "r") as f:
#         for line in f:
#             if "POST /v1" in line:
#                 post_count += 1
#             # After 2nd POST /v1, wait for "start forward..."
#             if post_count >= 2 and "start forward on the model" in line:
#                 if model_fwd_counts <= skip_model_fwd_counts: 
#                     model_fwd_counts += 1
#                     continue
#                 else:
#                     model_fwd_counts += 1
#                     if model_fwd_counts > skip_model_fwd_counts and not capture:
#                         print(f"Skip {skip_model_fwd_counts} model forwards and start capturing.")
#                         capture = True
#                         continue
             
#             # Stop at "stop forward..."
#             if capture and "stop forward on the model" in line:
#                 capture_counts += 1
#                 break
#             # Capture lines in between
#             if capture:
#                 # parse the line to get rank id, elapse time, start_layer and end_layer id
#                 per_layer_line = line.strip()
#                 print(f"Capture line: {per_layer_line}")
#                 match = re.search(pattern, per_layer_line)
#                 if match:
#                     rank_id = int(match.group(1))          # "0"
#                     elapsed_time = float(match.group(2))     # "3.848"
#                     start_layer = int(match.group(3))      # "0"
#                     end_layer = int(match.group(4))        # "15"
#                     layers = end_layer - start_layer
#                     if per_pp_rank_info[rank_id] is None:
#                         info = {'layers': layers, 'total_time': elapsed_time}
#                         per_pp_rank_info[rank_id] = info
#                     else:
#                         per_pp_rank_info[rank_id]["total_time"] += elapsed_time
#                 else:
#                     print("WARN: No match found.") 


if parse_log == "prompt":
    for fwd_id in fwd_list:
        parse_and_extract_target_lines(logfile, fwd_id, "prompt")
    
    print(f"Extracted lines: {captured_results.keys()}")
    # parse the extracted lines in the captured_results to get the target information
    target_results = []
    for fwd_id in fwd_list:
        print(f"Extract data of forward: {fwd_id}")
        result = parse_target_fwd_info(fwd_id, prompt_len, captured_results[fwd_id])
        target_results.append({'fwd_id': fwd_id, 'result': result})
    
    write_to_csv("prompt_result.csv", target_results)
    
    for item in target_results:
        fwd_id = item['fwd_id']
        per_pp_rank_info = item['result']
        print(f"Result of forward {fwd_id}")
        num_reqs, total_layers, global_total_time, global_avg_time \
            =cal_avg_time(per_pp_rank_info, PP_Ranks)
        
        for idx, info in per_pp_rank_info.items():
            print(f"Rank {idx} (unit:ms): layers={info['layers']} num_reqs={info['num_reqs']} avg={info['avg_time']:.3f} total={info['total_time']:.3f}")     
        print(f"Num of reqs: {num_reqs}")
        print(f"Total layers: {total_layers}")
        print(f"Global avg time (ms): {global_avg_time:.3f} ms")
        print(f"Global total time (ms): {global_total_time:.3f} ms")
    
elif parse_log == "decode":
    print(f"Parsing decode stage info...")
    for fwd_id in fwd_list:
        parse_and_extract_target_lines(logfile, fwd_id, "decode") 
    
    # calculate the average time per rank
    print(f"Extracted lines: {captured_results.keys()}")
    # parse the extracted lines in the captured_results to get the target information
    target_results = []
    for fwd_id in fwd_list:
        print(f"Extract data of forward: {fwd_id}")
        result = parse_target_fwd_info(fwd_id, 1, captured_results[fwd_id])
        target_results.append({'fwd_id': fwd_id, 'result': result})

    write_to_csv("decode_result.csv", target_results)
    
    for item in target_results:
        fwd_id = item['fwd_id']
        per_pp_rank_info = item['result']
        print(f"Result of forward {fwd_id}")
        num_reqs, total_layers, global_total_time, global_avg_time \
            =cal_avg_time(per_pp_rank_info, PP_Ranks)
        
        for idx, info in per_pp_rank_info.items():
            print(f"Rank {idx} (unit:ms): layers={info['layers']} num_reqs={info['num_reqs']} avg={info['avg_time']:.3f} total={info['total_time']:.3f}")     
        print(f"Num of reqs: {num_reqs}")
        print(f"Total layers: {total_layers}")
        print(f"Global avg time (ms): {global_avg_time:.3f} ms")
        print(f"Global total time (ms): {global_total_time:.3f} ms")
