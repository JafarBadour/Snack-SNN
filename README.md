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
# On windows
set TORCH_INCLUDE=C:\Users\BadourJ\Arts\Parallel-Dynamic-Sparse-Training\venv2\Lib\site-packages\torch\include\torch
```


```
python setup.py build

python setup.py install
```


connection from remote

```shell
ssh -L 8888:localhost:8888 -p 2222 jafar@16.62.171.242
```