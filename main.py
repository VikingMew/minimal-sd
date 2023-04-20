import torch

import modules.txt2img
import modules.sd_models
import modules.sd_vae
import modules.extensions


def main():
    # load all models
    modules.sd_models.list_models()
    modules.sd_vae.refresh_vae_list()
    modules.sd_models.load_model()
    modules.extensions.list_extensions()

    # run
    modules.txt2img.txt2img(
        "task(1)",
        "realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),",
        "EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,",
        [],
        20,  # steps
        16,  # samplerindex #todo
        True,
        False,
        1,
        1,
        7,
        1,  # seed
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
        args=(  # fuck
            0,
            "<scripts.external_code.ControlNetUnit object at 0x7f27c8e3ff10>",
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
        ),
    )


if __name__ == "__main__":
    main()
