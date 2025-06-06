# requires CUDA 12.6 + PyTorch==2.6
conda env create -f requirements.yaml

source /fyh/.env/miniconda3/etc/profile.d/conda.sh
conda activate vllm
which pip

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate einops sentencepiece tqdm requests opencv-python Pillow pandas datasets qwen_vl_utils -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm==0.8.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/