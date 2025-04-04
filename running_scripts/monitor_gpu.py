import time
import pynvml
import csv
import sys
import signal
import os

pynvml.nvmlInit()

device_id = 0
csv_file_path = "/home/jieye/viper2/benchmark_mii/perf_results/"
metric_values = []

def monitor_gpu(device_id, csv_file_path):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def signal_handler(sig, frame):
        print(f"====== Starting cleanup in signal_handler for GPU {device_id} ======")
        dump_to_csv()
        sys.exit(-1)

    def dump_to_csv():
        print(f"====== Starting to dump in CSV file. Num entries {len(metric_values)} for GPU {device_id} ======")
        try:
            with open(csv_file_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if (device_id == 0):
                    writer.writerow(["device_id", "timestamp", "txpci", "rxpci", "gpu_util", "mem_util", "mem_used", "mem_free"])
                    # writer.writerow(["device_id", "timestamp", "gpu_util", "mem_util", "mem_used", "mem_free"])
                writer.writerows(metric_values)
                csvfile.close()
                
            # Ensure its persisted
            f = open(csv_file_path, 'a+')
            os.fsync(f.fileno())
            f.close()
            print(f"====== Completed dump in CSV file. Num entries {len(metric_values)} for GPU {device_id} ======")
            pynvml.nvmlShutdown()
        except Exception as e:
            print('====== An exception occurred: {} ======'.format(e))

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            # Get PCIe throughput information
            throughput_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024  # in KB/s
            throughput_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024  # in KB/s
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if (throughput_rx + throughput_tx + utilization.gpu + utilization.memory) > 0:
                metric_values.append([device_id, time.time_ns(), throughput_tx, throughput_rx, utilization.gpu, utilization.memory, mem_info.used, mem_info.free])
            # if (utilization.gpu + utilization.memory) > 0:
            #     metric_values.append([device_id, time.time_ns(), utilization.gpu, utilization.memory, mem_info.used, mem_info.free])
            time.sleep(0.01)  # Adjust the sleep interval as needed
    except KeyboardInterrupt:
        print(f"====== Got interrupt in GPU monitoring script {device_id} ======")
        dump_to_csv()
    finally:
        print(f"====== Exiting monitoring script {device_id} ======")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <device_id> <csv_file_path>")
        sys.exit(1)

    device_id = int(sys.argv[1])
    csv_file_path = sys.argv[2]

    print(f"====== Starting monitoring script for GPU {device_id} ======")
    monitor_gpu(device_id, csv_file_path)
    print(f"====== Ending monitoring script for GPU {device_id} ======")