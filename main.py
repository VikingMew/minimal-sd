import asyncio
import hashlib
import io
import json
import logging
import random
import re
import subprocess
import time

import numpy as np
import torch

import modules.extensions
import modules.scripts
import modules.sd_models
import modules.sd_vae
import modules.txt2img

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s()] %(message)s",
)

"""
 {'self': <scripts.external_code.ControlNetUnit object at 0x7f3aa8c2ab50>, 'enabled': True, 'module': 'canny', 'model': 'control_canny-fp16 [e3fe7712]', 'weight': 1, 'image': {'image': array([[[0, 0, 0],
, 'invert_image': False, 'resize_mode': 'Scale to Fit (Inner Fit)', 'rgbbgr_mode': False, 'low_vram': False, 'processor_res': 512, 'threshold_a': 100, 'threshold_b': 200, 'guidance_start': 0, 'guidance_end': 0.22, 'guess_mode': False}   
"""
ARR_IMAGE = np.load("canny_image.npy")
ARR_MASK = np.load("canny_mask.npy")


def run_txt_img(positive, negative):
    # run
    control_net_unit = modules.scripts.scripts_data[0].script_class.get_unit(
        True,
        "canny",
        "control_canny-fp16 [e3fe7712]",
        1,
        dict(image=ARR_IMAGE, mask=ARR_MASK),
        "Scale to Fit (Inner Fit)",
        False,
        512,
        100,
        200,
        0,
        0.22,
        False,
    )
    return modules.txt2img.txt2img(
        "task(1)",
        positive,
        negative,
        [],
        20,  # steps
        16,  # samplerindex #todo
        True,
        False,
        1,
        1,
        7,
        random.randint(1, 1000000),  # seed
        -1,
        0,
        0,
        0,
        False,
        512,
        512,
        False,
        0.7,
        2,
        " hr_upscaler: str",
        0,
        0,
        0,
        [],
        0,
        control_net_unit,
        False,
        False,
        "positive",
        "comma",
        0,
        False,
        False,
        "",
        1,
        "",
        0,
        "",
        0,
        "",
        True,
        False,
        False,
        False,
        0,
        None,
        False,
        50,
    )


def make_app():
    return tornado.web.Application(
        [
            (r"/avatar", AvatarHandler),
        ]
    )


class JsonHandler(tornado.web.RequestHandler):
    async def get(self):
        self.write("Hello, world")

    async def prepare(self):
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = json.loads(self.request.body)
        else:
            self.json_args = None


class AvatarHandler(JsonHandler):
    async def post(self):
        data = self.json_args
        positive = data["postive"]
        negative = data["negative"]
        image = run_txt_img(positive, negative)[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        # set the HTTP headers and write the image data to the response
        self.set_header("Content-Type", "image/png")
        self.set_header("Content-Length", str(len(buffer.getvalue())))
        self.write(buffer.getvalue())


def main():
    # load all models
    modules.extensions.list_extensions()
    modules.scripts.load_scripts()
    modules.sd_models.list_models()
    modules.sd_vae.refresh_vae_list()
    modules.sd_models.load_model()
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    # print(
    #     run_txt_img(
    #         "realistic_portrait_female, (white background:1.5), (high detail skin:1.2),",
    #         "EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,",
    #     )[0][0]
    # )
    sockets = bind_sockets(8888)
    tornado.process.fork_processes(1)
    asyncio.run(post_fork_main(sockets))


async def post_fork_main(sockets):
    server = HTTPServer(make_app())
    server.add_sockets(sockets)
    await asyncio.Event().wait()


if __name__ == "__main__":
    main()
