wget https://gpu-1316675088.cos.ap-nanjing.myqcloud.com/sd/canny_mask.npy
wget https://gpu-1316675088.cos.ap-nanjing.myqcloud.com/sd/canny_image.npy
mkdir models
cd models && mkdir Stable-diffusion  && cd Stable-diffusion && wget https://gpu-1316675088.cos.ap-nanjing.myqcloud.com/sd/Stable-diffusion/new.safetensors
cd models && mkdir VAE && cd VAE && wget https://gpu-1316675088.cos.ap-nanjing.myqcloud.com/animevae.pt
mkdir extensions-builtin/sd-webui-controlnet/models && cd extensions-builtin/sd-webui-controlnet/models && wget https://gpu-1316675088.cos.ap-nanjing.myqcloud.com/control_canny-fp16.safetensors