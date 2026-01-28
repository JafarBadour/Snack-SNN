# Structure


```
├── benchmark
│   ├── gpu
│   │   └── profiler.py
│   ├── memory_leak
│   │   └── without_replacement.py
│   ├── plots
│   │   └── matrix_matrix.py
│   ├── pruner_grower
│   │   └── test_zeta_pruner_grower.py
│   └── sparse_matrix_multi
│       ├── __test_methods.py
│       ├── test_outer_join.py
│       ├── test.py
│       └── test_speed_sparse_tensor_vs_dense_tensor.py
├── DST
│   ├── initializers
│   │   ├── erdos.py
│   │   ├── fixed_degree.py
│   │   ├── grand.py
│   │   ├── __init__.py
│   │   └── uniform_initializer.py
│   ├── layers
│   │   ├── Dense.py
│   │   ├── __init__.py
│   │   └── Snack.py
│   └── pruner_grower
│       ├── __init__.py
│       └── zeta.py
├── sparse
│   └── mult
│       ├── csr
│       │   └── warp_csr.py
│       ├── setup.py
│       └── tensor
│           ├── __cls__.py
│           ├── exceptions.py
│           ├── __init__.py
│           └── utils.py
└── train
    └── DST
        ├── mnist.py
        └── train.py
```


There are mainly three experiemnt tracks in the repository
```
train/DST/train.py # results explained in figure 4 
train/DST/mnist.py # results not reported in the paper but it basically shows how to use SNACK in a real world training problem

benchmark/sparse_matrix_multi/test_speed_sparse_tensor_vs_dense_tensor.py # this reports Figure 3


```
to plot the results please find the necessary notebooks at
```
├── DST
│   └── plots and stats.ipynb # figure 4, [1-6] in Appendix
├── notebooks
│   ├── stats.ipynb  # Figure 3
│   └── testy.ipynb
└── train
    └── DST
        └── minst.ipynb # not reported in paper
```
# installing 

after installing nvcc and cuda 12.6 or 12.1

```
conda create -p ./venv2 python=3.12.7
```

run this 
```bash
sudo apt install nvidia-cuda-toolkit


```

check this link to install cuda 12.6

```bash
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local```
```
```
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


```
# On Windows, set TORCH_INCLUDE to your PyTorch installation path
# Example (modify for your username):
# set TORCH_INCLUDE=C:\Users\USERNAME\path\to\venv\Lib\site-packages\torch\include\torch
```


```
cd sparse/mult
python setup.py build
python setup.py install
```

# Set PyTorch library path (Linux)
# Modify the path according to your Python installation
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

if you have the dataset on a remote server you can connect from it using the following command

```shell
ssh -L 8888:localhost:8888 -p 2222 username@LINK
