* git clone  https://github.com/Stability-AI/stablediffusion.git ./repositories/stable-diffusion-stability-ai/
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
* git clone https://github.com/crowsonkb/k-diffusion.git ./repositories/k-diffusion
* pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
* cp ../stable-diffusion-webui/models/Stable-diffusion/new.safetensors ./models/Stable-diffusion/
* cp clip
* cp animaevae
* git clone https://github.com/Mikubill/sd-webui-controlnet.git ./extensions/sd-webui-controlnet/