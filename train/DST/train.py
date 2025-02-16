import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from DST.layers import Snack, Dense
from time import time as tic


def train(type_, sparsity=0):
    import random

    random.seed(42)
    input_size = 10000
    output_size = 10000
    if type_ == "Sparse":

        model = Snack(input_size, output_size, sparsity=sparsity).cuda()
    else:
        model = Dense(input_size, output_size).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Dummy data
    x = torch.randn((1, input_size)).cuda()
    target = torch.randn((1, output_size)).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    t1 = tic()
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
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    total_params = sum(p.numel() for p in model.parameters())
    # print("model params", total_params)
    # print(f"Execution time: {elapsed_time_ms:.6f} ms")
    return total_params, elapsed_time_ms


if __name__ == "__main__":
    data = []
    for rep in tqdm(list(range(1))):
        for sparsity in tqdm(np.linspace(0.5, 1, 10)):
            dense_total_params, dense_time = train(type_="Dense", sparsity=sparsity)
            sparse_total_params, sparse_time = train(type_="Sparse", sparsity=sparsity)
            data.append(
                {
                    "isSparse": "Sparse",
                    "cuda_elapsed_time": sparse_time,
                    "total_params": sparse_total_params,
                    "sparsity_level": sparsity,
                }
            )
            data.append(
                {
                    "isSparse": "Dense",
                    "cuda_elapsed_time": dense_time,
                    "total_params": dense_total_params,
                    "sparsity_level": sparsity,
                }
            )

    df = pd.DataFrame.from_records(data)
    df.to_csv("DST/log.csv", index=False)
