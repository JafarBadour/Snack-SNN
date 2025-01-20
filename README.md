# installing 

after installing nvcc and cuda 12.6

```
conda create -p ./venv2 python=3.12.7
```

run this 

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


```
set TORCH_INCLUDE=C:\Users\BadourJ\Arts\Parallel-Dynamic-Sparse-Training\venv2\Lib\site-packages\torch\include\torch
```


g++ -std=c++14 -I"C:/Users/BadourJ/Downloads/libtorch/libtorch-shared-with-deps-latest/libtorch/include" -I"C:/Users/BadourJ/Downloads/libtorch/libtorch-shared-with-deps-latest/libtorch/include/torch/csrc/api/include" \ main.cpp  -L"C:/Users/BadourJ/Downloads/libtorch/libtorch-shared-with-deps-latest/libtorch/lib"  -ltorch -lcaffe2 -o main.exe
