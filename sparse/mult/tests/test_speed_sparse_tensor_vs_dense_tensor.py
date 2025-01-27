from sparse.mult.tensor import SparseTensor
from tqdm import tqdm
from sparse.mult.tensor.utils import create_random_sparse_matrix
import time
import torch
import pandas as pd

sparsity_levels = (
         list(range(50, 80, 10)) +
         list(range(80, 96, 5)) +
        [96, 97, 98, 99])
# [5, 10, 15, ..., 90, 95, 96, 97, 98, 99]
# dense level is layer A with layer B that are after one another in the model architecture
dense_levels = {
    #  "500x500" : {'Reps' : 10},
    #  "1000x1000" : {'Reps' : 10},
    # "5000x5000" : {'Reps' : 10},
    "10000x10000" : {'Reps' : 10},
}

log = []

for dense_level in tqdm(dense_levels.keys(), desc="Processing Dense Levels"):
    layera, layerb = list(map(int, dense_level.split('x')))
    print(dense_level)
    for sparsity_level in tqdm(sparsity_levels, desc='Sparsity Lvls', leave=False):

        ones = torch.ones(layerb).cuda()

        sparse_matrix = create_random_sparse_matrix(layera, layerb, sparsity_level)

        # import ipdb;ipdb.set_trace()
        sparse_matrix = sparse_matrix.cuda()
        temp = sparse_matrix @ ones
        for rep in tqdm(list(range(dense_levels[dense_level]['Reps'])),desc='Reps sparse', leave=True):
            res=ones
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            t1 = time.time()
            for _ in range(100):
                res = sparse_matrix @ res
                res = res / res.max()

            s = res.sum()

            t2 = time.time()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            log.append({"isSparse" : "SparseUT", "dense_level" : dense_level,
                        "sparsity_level" : sparsity_level, "time" : t2 - t1, "cuda_elapsed_time": elapsed_time_ms, 'rep' : rep})

        dense_matrix = sparse_matrix.dense()
        print('finished sparse ops')
        for rep in tqdm(list(range(dense_levels[dense_level]['Reps'])), desc='Reps dense', leave=True):
            res = ones
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            t1 = time.time()
            for _ in range(100):
                res = res @ dense_matrix
                res = res / res.max()
            s = res.sum()

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            t2 = time.time()

            log.append({"isSparse": "Dense", "dense_level": dense_level,
                        "sparsity_level": sparsity_level, "time": t2 - t1, "cuda_elapsed_time": elapsed_time_ms,
                        'rep': rep})

        del dense_matrix
        indices = torch.concat((sparse_matrix.indices[:,0].reshape(1, -1),
                                sparse_matrix.indices[:,1].reshape(1, -1)), axis=0)

        sparse_tensor = torch.sparse_coo_tensor(indices, sparse_matrix.values, sparse_matrix.matrix_shape, device='cuda')
        del sparse_matrix
        torch.cuda.empty_cache()

        temp = ones @ sparse_tensor
        for rep in tqdm(list(range(dense_levels[dense_level]['Reps'])), desc='Reps sparse', leave=True):
            res = ones
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            t1 = time.time()
            for _ in range(100):
                res = sparse_tensor @ res
                res = res / res.max()

            s = res.sum()

            t2 = time.time()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)


            log.append({"isSparse" : "SparseTorch", "dense_level" : dense_level,
                        "sparsity_level" : sparsity_level, "time" : t2 - t1,  "cuda_elapsed_time": elapsed_time_ms, 'rep' : rep})
            df = pd.DataFrame(log)
            df.to_csv("./tests/log10k.csv", index=False)

        df = pd.DataFrame(log)
        df.to_csv("./tests/log10k.csv", index=False)
        del sparse_tensor


