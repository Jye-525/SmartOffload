import time
import csv
import sys
import signal
import os

import psutil

device_id = 0
csv_file_path = "/home/jieye/viper2/benchmark_mii/perf_results/"
metric_values = []

def monitor_host_mem(csv_file_path):

    def signal_handler(sig, frame):
        print(f"====== Starting cleanup in signal_handler for vmem monitor ======")
        dump_to_csv()
        sys.exit(-1)

    def dump_to_csv():
        print(f"====== Starting to dump in CSV file. Num entries {len(metric_values)} for vmem ======")
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp", "total_mem(MB)", "used_mem(MB)", "percent(%)"])
                writer.writerows(metric_values)
                csvfile.close()
                
            # Ensure its persisted
            f = open(csv_file_path, 'a+')
            os.fsync(f.fileno())
            f.close()
            print(f"====== Completed dump in CSV file. Num entries {len(metric_values)} for vmem ======")
        except Exception as e:
            print('====== An exception occurred: {} ======'.format(e))

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            # Get Host memory information
            vm_stats = psutil.virtual_memory()
            total_mem = round(((vm_stats.total) / (1024**2)), 2) # in MB
            used_mem = round(((vm_stats.total - vm_stats.available) / (1024**2)), 2)  # in MB
            percent = vm_stats.percent
            metric_values.append([time.time_ns(), total_mem, used_mem, percent])
            time.sleep(0.01)  # Adjust the sleep interval as needed
    except KeyboardInterrupt:
        print(f"====== Got interrupt in GPU monitoring script {device_id} ======")
        dump_to_csv()
    finally:
        print(f"====== Exiting monitoring script {device_id} ======")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    print(f"====== Starting monitoring script for vmem ======")
    monitor_host_mem(csv_file_path)
    print(f"====== Ending monitoring script for vmem ======")
