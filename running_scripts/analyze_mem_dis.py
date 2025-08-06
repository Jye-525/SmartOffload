import sys, os
import re
import csv
import pandas as pd

input_file = sys.argv[1]

def extract_info_from_filename(file_name):
    model_match = re.search(r"server_(.*?)_fixed", file_name)
    tp_pp_match = re.search(r"tp(\d+)_pp(\d+)", file_name)
    
    model_name = model_match.group(1) if model_match else "unknown"
    tp_value = tp_pp_match.group(1) if tp_pp_match else "0"
    pp_value = tp_pp_match.group(2) if tp_pp_match else "0"
    tp_pp = f"tp{tp_value}_pp{pp_value}"
    
    return model_name, tp_pp

def extract_mem_info(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else "N/A"

# Parse the log file to extract the target log information
def parse_log_file(input_file):
    target_logs = []
    with open(input_file, 'r') as f:
        for line in f:
            if "model weights take" in line:
                target_logs.append(line.strip())
    return target_logs

if __name__ == "__main__":
    input_file = sys.argv[1]
    print(f"Input file: {input_file}")
    assert input_file is not None, "Please provide a valid input file"
    # Get the directory path first
    base_path = os.path.dirname(input_file)  # "/home/jieye/moe_mix_precision/logs_Llama-3.1-405B_tp4_pp10/"
    file_name = os.path.basename(input_file)  # "server_Llama-3.1-405B_fixed_p1023_g1_r1_tp4_pp10_eager.log"
    
    # Extract the model name, tp, pp from the file name
    model_name, tp_pp = extract_info_from_filename(file_name)
    print(f"Model name: {model_name}, TP/PP: {tp_pp}")
     
    # Parse the input file and extract the target lines in the log
    target_logs = parse_log_file(input_file)
    if len(target_logs) == 0:
        print("No target log information found in the input file")
        sys.exit(-1)
    
    # Parse the target logs and save the results to the output CSV file
    patterns = {
        "PP_rank": r"PP Rank (\d+)",
        "TP_rank": r"TP Rank (\d+)",
        "model_weights": r"model weights take (\d+\.\d+)GiB",
        "non_torch": r"non_torch_memory takes (\d+\.\d+)GiB",
        "activation": r"PyTorch activation peak memory takes (\d+\.\d+)GiB",
        "max_kvcache_mem.": r"the rest of the memory reserved for KV Cache is (\d+\.\d+)GiB",
        "gpu_blocks": r"allocate (\d+) GPU blocks"
    } 
    
    # Output file name
    output_csv = f"{base_path}/memdist_{model_name}_{tp_pp}.csv"
    with open(output_csv, 'w', newline='') as csvfile:  # 'w' to overwrite, 'a' to append
        writer = csv.writer(csvfile)
        
        writer.writerow(patterns.keys())
        for log in target_logs:
            row = [extract_mem_info(log, p) for p in patterns.values()]
            writer.writerow(row)  # Write inside the loop, but file remains open
            
    # Print the csv data using pandas
    df = pd.read_csv(output_csv)
    # Sort by PP rank (ascending) and then TP rank (ascending)
    df_sorted = df.sort_values(by=["PP_rank", "TP_rank"], ascending=[True, True])
    print(df_sorted.to_string(index=False))
    df_sorted.to_csv(output_csv, index=False, sep=",")
    print(f"Analysis complete. Results saved to {output_csv}")