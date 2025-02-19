import time
import warnings

import pandas as pd
import multiprocessing as mp
import pynvml


class GPUAsyncProfiler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = None
        self.data = []
        self.start_time = None
        self.running = mp.Value("b", False)

    def _benchmark(self, running, interval, data):

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        cumulative_energy = 0.0
        while running.value:
            timestamp = time.time()
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_free = mem_info.free / (1024**2)
            memory_used = mem_info.used / (1024**2)
            total_memory = mem_info.total / (1024**2)
            cumulative_energy += power * (interval / 3600.0)
            data.append([timestamp, power, util, fan_speed, memory_free, memory_used, total_memory, cumulative_energy])

            time.sleep(interval)

        pynvml.nvmlShutdown()

    def start_benchmark(self):

        if self.process is None or not self.process.is_alive():
            self.data = mp.Manager().list()
            self.running.value = True
            self.process = mp.Process(target=self._benchmark, args=(self.running, self.interval, self.data))
            self.process.start()
        else:
            warnings.warn("Already benching, stopping the current benching job")
            self.stop_benchmark()
            return self.start_benchmark()

    def stop_benchmark(self):
        if self.process and self.process.is_alive():
            self.running.value = False
            self.process.join()
            self.process = None

            df = pd.DataFrame(
                list(self.data),
                columns=[
                    "Timestamp",
                    "Power (W)",
                    "Utilization (%)",
                    "Fan Speed (%)",
                    "Memory Free (MB)",
                    "Memory Used (MB)",
                    "Total Memory (MB)",
                    "Cumulative Energy (Wh)",
                ],
            )
            return df
        return None
