#!/usr/bin/python3
import whisper
import datetime
import time
import config
import os

# using third GPU. So that first one=0 will available to other projects by default.
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_NUMBER

options = {"beam_size": 5, "best_of": 5}


def log(message):
    current_time = datetime.now()

    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S') + '.' + str(current_time.microsecond // 1000).zfill(3)

    print(f'{formatted_time} - {message}')
    with open("client.log", "a") as file:
        file.write(f'{formatted_time} - {message}\n')

if __name__ == '__main__':
    log('start benchmark')
    whisper_model = whisper.load_model(config.whisper_model)
    log(f'whisper model {config.whisper_model} loaded')

    start_time = time.time()

    fastest_loop = 0
    slowest_loop = 0

    for i in range(config.iterations):
        one_loop_start_time = time.time()
        result = whisper_model.transcribe(config.file_path, **options)
        loop_time = time.time() - one_loop_start_time
        if loop_time < fastest_loop or fastest_loop == 0:
            fastest_loop = loop_time
        if loop_time > slowest_loop:
            slowest_loop = loop_time
        log(f'loop {i} time: {loop_time} sec')    
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000

    log(f'Total time: {total_time_ms} ms, average loop: {total_time_ms / config.iterations} ms, fastest loop: {fastest_loop} ms, slowest loop: {slowest_loop} ms')