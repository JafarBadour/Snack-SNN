import numpy as np
from tqdm import tqdm
import pandas as pd
from DST.layers import Snack, Dense
from benchmark.gpu.profiler import GPUAsyncProfiler
from DST.initializers.fixed_degree import FixedDegreeRandomInitializer

from time import time as tic


input_size = 10000
output_size = 10000


def train_n_batch_only_(input_size, output_size, batch_sz, type_, sparsity=0):
    import random
    import torch
    random.seed(42)
    t1 = tic()

    if type_ == "Snack":

        model = Snack(
            input_size,
            output_size,
            sparsity=sparsity,
            initializer=FixedDegreeRandomInitializer,
            debug=True,
        ).cuda()
    else:
        model = Dense(input_size, output_size).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Dummy data
    x = torch.randn((batch_sz, input_size)).cuda()
    target = torch.randn((batch_sz, output_size)).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    prof = GPUAsyncProfiler(0.05)
    prof.start_benchmark()
    start_event.record()

    torch.cuda.synchronize()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")
    # print(type_, tic() - t1)
    end_event.record()
    gpu_df = prof.stop_benchmark()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    total_params = sum(p.numel() for p in model.parameters())
    # print("model params", total_params)
    # print(f"Execution time: {elapsed_time_ms:.6f} ms")
    del model
    del x

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    del torch
    return total_params, elapsed_time_ms, gpu_df


def train_n_batch_only(output_file):
    data = []

    for rep in tqdm(list(range(1)), desc="Repeating"):
        for sparsity in tqdm(
            [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.985, 0.99, 0.995],
            desc="Sparsity processing",
        ):
            for batch_sz in tqdm([1, 2, 4, 8, 16, 32, 64, 128], desc="batch processing"):

                sparse_total_params, sparse_time, gpu_df = train_n_batch_only_(
                    batch_sz=batch_sz,
                    type_="Snack",
                    sparsity=sparsity,
                    input_size=input_size,
                    output_size=output_size,
                )
                data.append(
                    {
                        "isSparse": "Snack",
                        "batch_size": batch_sz,
                        "dense_level": f"{input_size}x{output_size}",
                        "cuda_elapsed_time": sparse_time,
                        "energy_spent_per_second": gpu_df["Cumulative Energy (Wh)"].iloc[-1] / sparse_time,
                        "total_params": sparse_total_params,
                        "sparsity_level": sparsity,
                        "rep": rep,
                        "energy_spent": gpu_df["Cumulative Energy (Wh)"].iloc[-1],
                        "average_power_used": gpu_df["Power (W)"].mean(),
                        "average_utilization": gpu_df["Utilization (%)"].mean(),
                        "average_memory_used": gpu_df["Memory Used (MB)"].mean(),
                    }
                )

                df = pd.DataFrame.from_records(data)
                df.to_csv(output_file, index=False)


    for rep in tqdm(list(range(1)), desc="Repeating"):
        for batch_sz in tqdm([1, 2, 4, 8, 16, 32, 64, 128], desc="batch processing"):
            for sparsity in tqdm(
                 [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
                    desc="Sparsity processing",
            ):
                dense_total_params, dense_time, gpu_df = train_n_batch_only_(
                    batch_sz=batch_sz,
                    type_="Dense+Mask",
                    input_size=input_size,
                    output_size=output_size,
                )
                data.append(
                    {
                        "isSparse": "Dense+Mask",
                        "batch_size": batch_sz,
                        "dense_level": f"{input_size}x{output_size}",
                        "cuda_elapsed_time": dense_time,
                        "total_params": dense_total_params,
                        "energy_spent_per_second": gpu_df["Cumulative Energy (Wh)"].iloc[-1] / dense_time,
                        "sparsity_level": sparsity,
                        "rep": rep,
                        "energy_spent": gpu_df["Cumulative Energy (Wh)"].iloc[-1],
                        "average_power_used": gpu_df["Power (W)"].mean(),
                        "average_utilization": gpu_df["Utilization (%)"].mean(),
                        "average_memory_used": gpu_df["Memory Used (MB)"].mean(),
                    }
                )


                df = pd.DataFrame.from_records(data)
                df.to_csv(output_file, index=False)



if __name__ == "__main__":
    train_n_batch_only(output_file="DST/log2.csv")
