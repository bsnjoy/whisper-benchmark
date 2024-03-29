#!/usr/bin/env python3
import whisper
from datetime import datetime
import time
import config
import os
import sys
import platform
import distro
import GPUtil

# Check if at least one additional argument is provided
if len(sys.argv) > 1:
    gpu_num = sys.argv[1]
    print(f"Set GPU Number from command line: {gpu_num}")
else:
    gpu_num = config.GPU_NUMBER

# Which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Construct the absolute path to the file
audio_file_path = os.path.join(script_dir, config.file_path)


options = {"beam_size": 5, "best_of": 5}

def get_os_info():
        # Check if the operating system is Linux
    if platform.system() == "Linux":
        # Get distribution name and version
        os_info = f'{distro.name()} {distro.version()}'
    else:
        # For non-Linux operating systems
        os_info = f'{platform.platform()}'
    return os_info

def log(message):
    current_time = datetime.now()

    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S') + '.' + str(current_time.microsecond // 1000).zfill(3)

    print(f'{formatted_time} - {message}')
    with open("client.log", "a") as file:
        file.write(f'{formatted_time} - {message}\n')

if __name__ == '__main__':
    log(f'Loading whisper model {config.whisper_model} in GPU {gpu_num}.... wait...')
    start_total_time = time.time()
    start_model_time = time.time()
    whisper_model = whisper.load_model(config.whisper_model)
    whisper_load_time = time.time() - start_model_time
    log(f'whisper model {config.whisper_model} loaded in {whisper_load_time:.3f} sec')

    start_transcribe_time = time.time()

    fastest_loop = 0
    slowest_loop = 0

    for i in range(config.iterations):
        one_loop_start_time = time.time()
        result = whisper_model.transcribe(audio_file_path, **options)
        loop_time = time.time() - one_loop_start_time
        if loop_time < fastest_loop or fastest_loop == 0:
            fastest_loop = loop_time
        if loop_time > slowest_loop:
            slowest_loop = loop_time
        log(f'loop {i} time: {loop_time} sec. result: {result["text"]}')
    end_time = time.time()
    
    transcribe_time = (end_time - start_transcribe_time)
    total_time = (end_time - start_total_time)

    gpus = GPUtil.getGPUs()
    gpu = gpus[int(gpu_num)]
    print('| OS | GPU | GPU ID | GPU Load | GPU Free | GPU Used | Gpu Temp | Total, sec | Model | Model Load, sec | Transcribe, sec | Average Loop, sec | Fastest Loop Time | Slowest Loop, sec |')
    print(f'| {get_os_info()} | {gpu.name} | {gpu_num} | {gpu.load*100}% | {gpu.memoryFree}MB | {gpu.memoryUsed}MB | {gpu.temperature} °C | {total_time:.3f} | {config.whisper_model} | {whisper_load_time:.3f} | {transcribe_time:.3f} | {total_time / config.iterations:.3f} | {fastest_loop:.3f} | {slowest_loop:.3f} |')
    print(f'{get_os_info()} {gpu.name} GPU: {gpu_num}, Total: {total_time:.3f} sec, Model {config.whisper_model} load: {whisper_load_time:.3f} sec, Transcribe: {transcribe_time:.3f} sec, average loop: {total_time / config.iterations:.3f} sec, fastest: {fastest_loop:.3f} sec, slowest: {slowest_loop:.3f} sec')
