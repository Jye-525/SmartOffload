import re, sys, csv, os
from typing import List, Dict, Tuple
from collections import defaultdict

############### Extract the parameters size of each layer from the log file ###############
def extract_parameters_per_layer(file_name) -> List[Dict[str, any]]:
    """
    Extracts parameters per layer from the log file.
    """
    target_pattern = r"Layer 0 with parameter name:\s*([^,]+),\s*weights_tensor_shape:\s*torch\.Size\(([^)]+)\),\s*weights_size:\s*([\d.]+)\s*MB" 
    target_results = []
    
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith("(RayWorkerWrapper"):
                # Skip lines starting with "(RayWorkerWrapper"
                continue
            if "init engine (profile, create kv cache, warmup model) took" in line:
                break
            match = re.search(target_pattern, line)
            if match:
                # Extract the layer name and parameters
                layer_name = match.group(1)
                weights_tensor_shape = match.group(2)
                weights_size = float(match.group(3))

                param_info = {
                    'layer_name': layer_name,
                    'weights_tensor_shape': weights_tensor_shape,
                    'weights_size(MB)': float(weights_size)
                }
                target_results.append(param_info)
          
    return target_results
    
def write_layer_parameters_to_csv(layer_parameters: List[Dict[str, any]], output_csv_file: str):
    """
    Write the extracted layer parameters to a CSV file.
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['layer_name', 'weights_tensor_shape', 'weights_size(MB)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for layer in layer_parameters:
            writer.writerow(layer)
            

############### Extract memory distribution of the model across each PP rank and TP rank ###############
def Extract_mem_distribution(file_name):
    """
    Extracts memory distribution from the log file.
    """
    # Define the regex pattern to match the memory distribution lines
    # (RayWorkerWrapper pid=3918397) INFO 03-29 01:24:36 worker.py:272] model weights take 15.63GiB; non_torch_memory takes 0.81GiB; PyTorch activation peak memory takes 3.52GiB; the rest of the memory reserved for KV Cache is 11.57GiB.The engine can allocate 12225 GPU blocks, 4228 CPU blocks on PP Rank 1 TP Rank 0.
    target_pattern = r"model weights take (\d+\.\d+)GiB; non_torch_memory takes (\d+\.\d+)GiB; PyTorch activation peak memory takes (\d+\.\d+)GiB; the rest of the memory reserved for KV Cache is (\d+\.\d+)GiB.The engine can allocate (\d+) GPU blocks, (\d+) CPU blocks on PP Rank (\d+) TP Rank (\d+)."
    target_results = []
    
    with open(file_name, "r") as f:
        for line in f:
            if "model weights take" not in line:
                # Skip lines if not containing "model weights take"
                continue
            if "init engine (profile, create kv cache, warmup model) took" in line:
                break
            match = re.search(target_pattern, line)
            if match:
                # Extract the memory distribution information
                model_weights = float(match.group(1))
                non_torch_memory = float(match.group(2))
                activation_memory = float(match.group(3))
                max_kvcache_memory = float(match.group(4))
                gpu_blocks = int(match.group(5))
                # cpu_blocks = match.group(6)
                pp_rank = int(match.group(7))
                tp_rank = int(match.group(8))

                param_info = {
                    'PP_rank': pp_rank,
                    'TP_rank': tp_rank,
                    'model_weights': model_weights,
                    'non_torch_mem': non_torch_memory,
                    'activation_mem': activation_memory,
                    'max_kvcache_mem': max_kvcache_memory,
                    'gpu_blocks': gpu_blocks,
                }
                target_results.append(param_info)
    
    sorted_results = sorted(target_results, key=lambda x: (x['PP_rank'], x['TP_rank']))      
    return sorted_results

def write_mem_results_to_csv(mem_results, output_csv_file):
    """
    Write the memory distribution results to a CSV file.
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['PP_rank', 'TP_rank', 'model_weights', 'non_torch_mem', 'activation_mem', 'max_kvcache_mem', 'gpu_blocks']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in mem_results:
            writer.writerow(result)
    print(f"Memory distribution results written to {output_csv_file}")
    
    
################ Extract the number of layers and start/end layer id on per PP Rank from the log file ###############
def Extract_per_PP_Rank_layer(file_name) -> Dict[int, Tuple[int, int, int]]:
    """
    Extracts the number of layers and start/end layer id on per PP Rank from the log file.
    For returned Dict, the key is the PP Rank.
    """
    target_pattern = r"PP Rank (\d+).*?TP Rank 0 include\s*(\d+).*?start layer:\s*(\d+)\s*end layer:\s*(\d+)"
    target_results = {}
    
    with open(file_name, "r") as f:
        for line in f:
            if "init engine (profile, create kv cache, warmup model) took" in line:
                break
            match = re.search(target_pattern, line)
            if match:
                pp_rank = int(match.group(1))
                num_layers = int(match.group(2))
                start_layer = int(match.group(3))
                end_layer = int(match.group(4))

                target_results[pp_rank] = (num_layers, start_layer, end_layer)
          
    return target_results

def write_layers_per_RR_rank_to_csv(layer_info: Dict[int, Tuple[int, int, int]], output_csv_file: str):
    """
    Write the extracted layer information to a CSV file.
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['PP_rank', 'num_layers', 'start_layer', 'end_layer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for pp_rank, (num_layers, start_layer, end_layer) in layer_info.items():
            writer.writerow({
                'PP_rank': pp_rank,
                'num_layers': num_layers,
                'start_layer': start_layer,
                'end_layer': end_layer
            })
    print(f"Layer information results written to {output_csv_file}")           
    
################# Extract the per model forward duration based on the given forward ids ###############
# Extract the per model forward duration from the log file
def Extract_forward_duration(file_name, fwd_ids: List[int], parse_type="prompt"):
    """
    Extracts the forward duration from the log file.
    """
    
    start_fwd_pattern = r"\[abs_timestamp\(ns\): (\d+)\].*?\[relative_start\(ns\): ([0-9.]+)\].*?PP Rank\s*(\d+).*?TP Rank\s*(\d+).*?start forward on the model, fwd_counts:\s*(\d+).*?input_ids_shape:\s*(\d+)"
    stop_fwd_pattern = r"\[abs_timestamp\(ns\): (\d+)\].*?\[relative_end\(ns\): ([0-9.]+)\].*?PP Rank\s*(\d+).*?TP Rank\s*(\d+).*?stop forward on the model. fwd_counts:\s*(\d+).*?input_ids_shape:\s*(\d+)"
    
    target_results = {}
    # Open the log file and get the target lines
    with open(file_name, "r") as f:
        for line in f:
            if "fwd_counts" not in line:
                continue
            if "fwd_counts: 1 input_ids_shape" in line:
                continue
            
            if "start forward on the model" in line:
                # parse the start forward info
                match_start_fwd = re.search(start_fwd_pattern, line.strip())
                if match_start_fwd:
                    pp_rank = int(match_start_fwd.group(3))
                    tp_rank = int(match_start_fwd.group(4))
                    abs_timestamp = int(match_start_fwd.group(1))
                    relative_start = float(match_start_fwd.group(2))
                    fwd_id = int(match_start_fwd.group(5))
                    input_ids_shape = int(match_start_fwd.group(6))
                    
                    if fwd_id in fwd_ids:
                        start_info = {
                            'pp_rank': pp_rank,
                            'tp_rank': tp_rank,
                            'abs_timestamp': abs_timestamp,
                            'relative_start': relative_start,
                            'input_shape': input_ids_shape
                            }
                        if fwd_id not in target_results:
                            target_results[fwd_id] = {"start_info": [start_info]}
                        else:
                            if "start_info" not in target_results[fwd_id]:
                                target_results[fwd_id]["start_info"] = [start_info]
                            else:
                                target_results[fwd_id]["start_info"].append(start_info)  
                continue
            
            if "stop forward on the model" in line:
                # parse the stop forward info
                match_stop_fwd = re.search(stop_fwd_pattern, line.strip())
                if match_stop_fwd:
                    pp_rank = int(match_stop_fwd.group(3))
                    tp_rank = int(match_stop_fwd.group(4))
                    abs_timestamp = int(match_stop_fwd.group(1))
                    relative_end = float(match_stop_fwd.group(2))
                    fwd_id = int(match_stop_fwd.group(5))
                    input_ids_shape = int(match_stop_fwd.group(6))
                    if fwd_id in fwd_ids:
                        stop_info = {
                            'pp_rank': pp_rank,
                            'tp_rank': tp_rank,
                            'abs_timestamp': abs_timestamp,
                            'relative_end': relative_end,
                            'input_shape': input_ids_shape
                            }
                       
                        if fwd_id not in target_results:
                            target_results[fwd_id] = {"stop_info": [stop_info]}
                        else:
                            if "stop_info" not in target_results[fwd_id]:
                                target_results[fwd_id]["stop_info"] = [stop_info]
                            else:
                                target_results[fwd_id]["stop_info"].append(stop_info) 
                        # calculate the elapsed time
    
    print(target_results)
    
    result = {}
    # Parse the target_results and calculate the elapsed time for each fwd_id
    for fwd_id, value in target_results.items():
        start_info = value['start_info']
        stop_info = value['stop_info']
        
        assert len(start_info) == len(stop_info), f"Mismatch in start and stop info for fwd_id {key}"
        tp_ranks = len(start_info)
        
        # Calculate durations for each tp_rank
        result[fwd_id] = {}
        for tp_rank in range(tp_ranks):
            relative_start = start_info[tp_rank]['relative_start']
            relative_end = stop_info[tp_rank]['relative_end']
            abs_start = start_info[tp_rank]['abs_timestamp']
            abs_end = stop_info[tp_rank]['abs_timestamp']
            
            relative_dur = (relative_end - relative_start) / 1e6  # Convert ns to ms
            abs_dur = (abs_end - abs_start) / 1e6  # Convert ns to ms
            input_shape = start_info[tp_rank]['input_shape']
            result[fwd_id][tp_rank] = {
                'relative_duration': relative_dur,
                'abs_duration': abs_dur,
                'input_shape': input_shape
            } 
    
    # get the max duration for each fwd_id, max value of all tp_ranks
    processed_result = {}
    for fwd_id, value in result.items():
        # Find the tp_rank with the maximum relative_duration
        max_tp_rank = max(
            value.keys(),
            key=lambda tp_rank: value[tp_rank]['relative_duration']
        )
        
        # Get both durations from the same tp_rank
        max_relative_dur = value[max_tp_rank]['relative_duration']
        max_abs_dur = value[max_tp_rank]['abs_duration']
    
        processed_result[fwd_id] = {
            'relative_duration': max_relative_dur,
            'abs_duration': max_abs_dur,
            'input_shape': value[0]['input_shape']
        }
    return processed_result

def write_forward_duration_to_csv(forward_duration: Dict[int, Dict[str, float]], output_csv_file: str, include_dur_1: bool = False):
    """
    Write the extracted forward duration to a CSV file.
    """
    with open(output_csv_file, mode='w', newline='') as csvfile:
        if include_dur_1:
            fieldnames = ['fwd_id', 'duration(ms)', 'duration_1(ms)', 'input_shape']
        else:
            fieldnames = ['fwd_id', 'duration(ms)', 'input_shape']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for fwd_id, durations in forward_duration.items():
            if include_dur_1:
                writer.writerow({
                    'fwd_id': fwd_id,
                    'duration(ms)': durations['relative_duration'],
                    'duration_1(ms)': durations['abs_duration'],
                    'input_shape': durations['input_shape']
                })
            else:
                writer.writerow({
                    'fwd_id': fwd_id,
                    'duration(ms)': durations['relative_duration'],
                    'input_shape': durations['input_shape']
                })
    print(f"Forward duration results written to {output_csv_file}")
    
################ Extract the per layer duration based on the given fwd ids ###############
def Extract_per_layer_fwd_time(file_name, fwd_ids: List[int], parse_type="prompt"):
    """
    Extracts the per-layer forward time from the log file.
    """
    layer_fwd_pattern = r"PP Rank\s*(\d+).*?TP Rank\s*(\d+).*?Layer\s*(\d+).*?relative timing info, layer_relative_start:\s*(\d+\.\d+).*?layer_relative_end:\s*(\d+\.\d+).*?layer_fwd_time:\s*(\d+\.\d+).*?ms, fwd_counts:\s*(\d+).*?input_ids_shape:\s*(\d+).*?num_prefill_reqs:\s*(\d+).*?num_decode_reqs:\s*(\d+)"
    
    layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    # Open the log file and get the target lines
    with open(file_name, "r") as f:
        for line in f:
            if "fwd_counts" not in line:
                continue
            # if "fwd_counts: 1 input_ids_shape" in line:
            #     continue
            
            match_layer_fwd = re.search(layer_fwd_pattern, line.strip())
            if match_layer_fwd:
                pp_rank = int(match_layer_fwd.group(1))
                tp_rank = int(match_layer_fwd.group(2))
                Layer_id = int(match_layer_fwd.group(3))
                relative_start = float(match_layer_fwd.group(4))
                relative_end = float(match_layer_fwd.group(5))
                layer_dur_1 = float(match_layer_fwd.group(6))
                fwd_id = int(match_layer_fwd.group(7))
                input_ids_shape = int(match_layer_fwd.group(8))
                num_prefill_reqs = int(match_layer_fwd.group(9))
                num_decode_reqs = int(match_layer_fwd.group(10))
                
                if fwd_id in fwd_ids:
                    layer_info = {
                        'pp_rank': pp_rank,
                        'tp_rank': tp_rank,
                        'layer_id': Layer_id,
                        'layer_relative_start': relative_start,
                        'layer_relative_end': relative_end,
                        'layer_dur_1': layer_dur_1,
                        'input_shape': input_ids_shape,
                        'num_prefill_reqs': num_prefill_reqs,
                        'num_decode_reqs': num_decode_reqs
                        } 
                    layer_data[fwd_id][pp_rank][tp_rank][Layer_id] = layer_info
                      
    return layer_data

def process_layer_data(layer_data: Dict[int, Dict[int, Dict[int, Dict[int, Dict[str, any]]]]], num_total_layers: int = 32) -> Dict[int, Dict[str, float]]:
    """
    Process the layer data to extract relevant information.
    """
    # Process the layer data to calculate the layer duration
    layer_info = defaultdict(lambda: defaultdict(dict))
    for fwd_id, value in layer_data.items():
        for pp_rank, pp_value in value.items(): 
            for tp_rank, tp_value in pp_value.items():
                print(f"PP Rank: {pp_rank} TP Rank: {tp_rank}, value: {len(tp_value)}")
                for layer_id in tp_value.keys():
                    # assert layer_id in range(pp_start_layer, pp_end_layer), f"Layer ID {layer_id} is out of range for PP Rank {pp_rank} and TP Rank {tp_rank}"
                    layer_relative_start = tp_value[layer_id]['layer_relative_start']
                    layer_relative_end = tp_value[layer_id]['layer_relative_end']
                    layer_dur_1 = tp_value[layer_id]['layer_dur_1']
                    if layer_id == 0:
                        layer_value = {
                            'layer_relative_start': layer_relative_start,
                            'layer_relative_end': layer_relative_end,
                            'layer_dur_1': layer_dur_1
                        }
                        layer_info[fwd_id][tp_rank][layer_id] = layer_value
                    elif layer_id < num_total_layers - 1:
                        # not the layer
                        layer_value = {
                            'layer_relative_start': layer_relative_start,
                            'layer_relative_end': layer_relative_end,
                            'layer_dur_1': layer_dur_1
                        }
                        layer_info[fwd_id][tp_rank][layer_id] = layer_value
                        # the layer_duration should be the difference between the current layer (layer_relative_start) and previous layer (layer_relative_start)
                        prev_layer_id = layer_id - 1
                        previous_layer_dur = layer_relative_start - layer_info[fwd_id][tp_rank][prev_layer_id]['layer_relative_start']
                        # update the layer_info to add the previous layer duration
                        layer_info[fwd_id][tp_rank][prev_layer_id]['layer_dur'] = previous_layer_dur
                        if layer_id == 13 or layer_id == 26 or layer_id == 39 or layer_id == 52:
                            print(f"layer_id: {layer_id}, previous_layer_dur: {previous_layer_dur}, layer_relative_start: {layer_relative_start}, previous_layer_relative_start: {layer_info[fwd_id][tp_rank][prev_layer_id]['layer_relative_start']}")
                    elif layer_id == num_total_layers - 1:
                        # last layer
                        layer_value = {
                            'layer_relative_start': layer_relative_start,
                            'layer_relative_end': layer_relative_end,
                            'layer_dur_1': layer_dur_1
                        }
                        layer_info[fwd_id][tp_rank][layer_id] = layer_value
                        # Update previous layer duration
                        prev_layer_id = layer_id - 1
                        previous_layer_dur = layer_relative_start - layer_info[fwd_id][tp_rank][prev_layer_id]['layer_relative_start']
                        layer_info[fwd_id][tp_rank][prev_layer_id]['layer_dur'] = previous_layer_dur
                        # Update the last layer duration 
                        cur_layer_dur = layer_relative_end - layer_relative_start
                        layer_value['layer_dur'] = cur_layer_dur 
                        layer_info[fwd_id][tp_rank][layer_id] = layer_value                      

    # Group all `layer_info` values by `layer_id` for each fwd_id
    # grouped_layer_info = defaultdict(list)
    grouped_layer_info = defaultdict(lambda: defaultdict(list))
    for fwd_id, value in layer_info.items():
        for tp_rank_id, tp_rank_value in value.items():
            for layer_id, layer_value in tp_rank_value.items():
                layer_dur_value = {
                    'layer_dur': layer_value['layer_dur'],
                    'layer_dur_1': layer_value['layer_dur_1']
                }
                grouped_layer_info[fwd_id][layer_id].append(layer_dur_value)
            
    # Get the max layer duration among the ranks for each layer
    processed_layer_info = defaultdict(dict)
    for fwd_id, value in grouped_layer_info.items():
        for layer_id, layer_value in value.items():
            # get the idx with max layer_dur
            max_idx = max(range(len(layer_value)), key=lambda i: layer_value[i]['layer_dur'])
            processed_layer_info[fwd_id][layer_id] = layer_value[max_idx] 
        
    return processed_layer_info
 
# Write the processed layer info to a CSV file
def write_layer_info_to_csv(layer_info, file_name, include_dur_1=False):
    """
    Write the layer info results to a CSV file.
    """
    with open(file_name, mode='w', newline='') as csvfile:
        if include_dur_1:
            fieldnames = ['layer_id', 'duration(ms)', 'duration_1(ms)']
        else:
            fieldnames = ['layer_id', 'duration(ms)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for layer_id, layer_value in layer_info.items():
            print(f"layer_id: {layer_id}, layer_value: {layer_value}")
            if include_dur_1:
                writer.writerow({'layer_id': layer_id, 'duration(ms)': layer_value['layer_dur'], 'duration_1(ms)': layer_value['layer_dur_1']})
            else:
                writer.writerow({'layer_id': layer_id, 'duration(ms)': layer_value['layer_dur']})
    print(f"Layer info results written to {file_name}")



if __name__ == "__main__":
    # get the parameters from the command line
    logfile = sys.argv[1]
    fwd_ids = list(map(int, sys.argv[2].split(",")))
    
    # Extract model name, PP and TP rank, prompt_length, gen_length, batch size, gpu_mem from the log file
    base_dir = os.path.dirname(logfile)
    file_name=os.path.basename(logfile)
    # Split into parts
    parts = file_name.split('_')

    # Extract fields
    model_name = parts[1]  # "deepseek-coder-33b-base"
    prompt_len = int(parts[3].replace('p', ''))  # 32767 (from "p32767")
    gen_len = int(parts[4].replace('g', ''))  # 1 (from "g1")
    requests = int(parts[5].replace('r', ''))  # 1 (from "r1")
    tp_ranks = int(parts[6].replace('tp', ''))  # 2 (from "tp2")
    pp_ranks = int(parts[7].replace('pp', ''))  # 2 (from "pp2")
    gpu = float(parts[8].replace('gpu', ''))  # 0.8 (from "gpu0.8")

    print({
        "model_name": model_name,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "requests": requests,
        "tp_ranks": tp_ranks,
        "pp_ranks": pp_ranks,
        "gpu": gpu
    }) 
    
    # get the base_csv_file_name
    base_csv_dir= os.path.join(base_dir, "csv_results")
    os.makedirs(base_csv_dir, exist_ok=True)
    base_csv_file_prefix= f"{model_name}_prompt{prompt_len}_gen{gen_len}_requests{requests}_tp{tp_ranks}_pp{pp_ranks}_gpu{gpu}"
    
    # Extract the parameters per layer from the log file
    layer_parameters = extract_parameters_per_layer(logfile)
    # Write the layer parameters to a CSV file
    layer_parameters_csv_file = os.path.join(base_csv_dir, f"{base_csv_file_prefix}_layer_parameters.csv")
    write_layer_parameters_to_csv(layer_parameters, layer_parameters_csv_file)
    
    # Extract the memory distribution from the log file
    mem_distribution = Extract_mem_distribution(logfile)
    # Write the memory distribution to a CSV file
    mem_distribution_csv_file = os.path.join(base_csv_dir, f"{base_csv_file_prefix}_mem_distribution.csv")
    write_mem_results_to_csv(mem_distribution, mem_distribution_csv_file)
    # Extract the number of layers and start/end layer id on per PP Rank from the log file
    
    # Extract the number of layers and start/end layer id on per PP Rank from the log file
    per_PP_rank_layers = Extract_per_PP_Rank_layer(logfile)
    # Write the layer info to a CSV file
    per_PP_rank_layers_csv_file = os.path.join(base_csv_dir, f"{base_csv_file_prefix}_layers_per_PP_rank.csv")
    write_layers_per_RR_rank_to_csv(per_PP_rank_layers, per_PP_rank_layers_csv_file)
    
    
    # Extract the forward duration from the log file
    forward_duration = Extract_forward_duration(logfile, fwd_ids)
    # Write the forward duration to a CSV file
    forward_duration_csv_file = os.path.join(base_csv_dir, f"{base_csv_file_prefix}_forward_dur.csv") 
    write_forward_duration_to_csv(forward_duration, forward_duration_csv_file, include_dur_1=True)
    
    
    # Extract the per-layer forward time from the log file
    layer_data = Extract_per_layer_fwd_time(logfile, fwd_ids)
    # Process the layer data to extract relevant information
    processed_layer_info = process_layer_data(layer_data, num_total_layers=per_PP_rank_layers[0][0])
    # Write the processed layer info to a CSV file
    for fwd_id, fwd_value in processed_layer_info.items():
        layer_info_csv_file = os.path.join(base_csv_dir, f"{base_csv_file_prefix}_layer_info_fwd{fwd_id}.csv")
        write_layer_info_to_csv(fwd_value, layer_info_csv_file, include_dur_1=True)
