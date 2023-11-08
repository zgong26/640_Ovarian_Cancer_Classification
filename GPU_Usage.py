import subprocess
import time


def get_gpu_usage():
    # Run the nvidia-smi command
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Extract the utilization percentage
    gpu_utilization = result.stdout.strip()

    return gpu_utilization


while True:
    # Get current GPU usage
    usage = get_gpu_usage()

    # Print the GPU usage
    print(f"Current GPU usage is: {usage}%")

    # Wait for 5 seconds
    time.sleep(5)
