FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /workspace

RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    diffusers==0.33.1 \
    accelerate==1.6.0 \
    peft==0.15.2 \
    timm==1.0.15 \
    einops==0.8.1 \
    safetensors==0.5.3 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.15.2 \
    decord==0.6.0 \
    pillow==11.1.0 \
    matplotlib==3.10.1 \
    tqdm==4.67.1 \
    pyyaml==6.0.2 \
    torcheval==0.0.7 \
    jupyterlab==4.3.6 \
    lpips==0.1.4 \
    dreamsim==0.2.1