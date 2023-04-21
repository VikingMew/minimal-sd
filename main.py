import torch
import numpy as np
import modules.extensions
import modules.scripts
import modules.sd_models
import modules.sd_vae
import modules.txt2img

"""
 {'self': <scripts.external_code.ControlNetUnit object at 0x7f3aa8c2ab50>, 'enabled': True, 'module': 'canny', 'model': 'control_canny-fp16 [e3fe7712]', 'weight': 1, 'image': {'image': array([[[0, 0, 0],
, 'invert_image': False, 'resize_mode': 'Scale to Fit (Inner Fit)', 'rgbbgr_mode': False, 'low_vram': False, 'processor_res': 512, 'threshold_a': 100, 'threshold_b': 200, 'guidance_start': 0, 'guidance_end': 0.22, 'guess_mode': False}   
"""


a = (
    "task(ewnahhbr5assnfo)",
    "realistic_portrait_male, (white background:1.5), (deep scheming:1.3), (thick hair:1.5), cold and overbearing, have a handsome face, Elf ear, (Golden eyes:1.3), thin lip, cocked eyebrow, high bridge of nose, V-shaped eyebrows,\n",
    "EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms, legs), extra limbs, long neck, signature, watermark, name,",
    [],
    31,
    16,
    True,
    False,
    1,
    1,
    7,
    659341459.0,
    -1.0,
    0,
    0,
    0,
    False,
    384,
    384,
    False,
    0.7,
    2,
    "Latent",
    0,
    0,
    0,
    [],
    0,
    "<scripts.external_code.ControlNetUnit object at 0x7f8fe0ba8e50>",
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


def main():
    # load all models
    modules.extensions.list_extensions()
    modules.scripts.load_scripts()
    modules.sd_models.list_models()
    modules.sd_vae.refresh_vae_list()
    modules.sd_models.load_model()
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    arr_image = np.load("canny_image.npy")
    arr_mask = np.load("canny_mask.npy")
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
        0,
        modules.scripts.scripts_data[0].script_class.get_unit(
            True,
            "canny",
            "control_canny-fp16 [e3fe7712]",
            1,
            dict(image=arr_image, mask=arr_mask),
            "Scale to Fit (Inner Fit)",
            False,
            512,
            100,
            200,
            0,
            0.22,
            False,
        ),
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


if __name__ == "__main__":
    main()
