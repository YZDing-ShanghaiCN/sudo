# sudo： My robot work in Sudo company during 2026 in Shanghai

# 1. pose

This directory is for object pose estimation from RGB image in robot manipulation tasks. 

--  ENVIRONMENT SETUP ------------- RTX 5060 8GB

reference link: [CSDN](https://blog.csdn.net/2504_93649063/article/details/159248787?spm=1001.2014.3001.5501) [CSDN](https://blog.csdn.net/2504_93649063/article/details/158879459?spm=1001.2014.3001.5501)

``` bash
# cuda preperation
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
python - <<'PY'
import torch
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Available:", torch.cuda.is_available())
print("GPU sm type:", torch.cuda.get_device_capability(0))
PY

# dependency installation
pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST="12.0"
pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

export all_proxy=socks5://127.0.0.1:7897
export ALL_PROXY=socks5://127.0.0.1:7897
export PYTHONPATH=$PYTHONPATH:/home/user/Desktop/main/posemain/segment_anything

python scripts/project.py
```
