import torch
from sparse_tensor_multiply import random_init_without_replacement
import time
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


input_size = 40000
output_size = 40000
obi = random_init_without_replacement(
    input_size,
    output_size,
    1 * input_size * output_size // 10,
    torch.empty((0, 2), dtype=torch.int32),
)
print("waiting while having an object")
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
print(obi.sum())
time.sleep(1)
del obi
print("finito waiting 10 secs and exiting")
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
time.sleep(1)
