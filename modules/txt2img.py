import modules.scripts
import modules.shared as shared
from modules import sd_samplers
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    process_images,
)
from modules.shared import cmd_opts, opts

# import modules.processing as processing
# from modules.ui import plaintext_to_html


"""  {'id_task': 'task(60ejhyv0rs1xfit)',
 'prompt': 'realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),', 
 'negative_prompt': 'EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,', 
 'prompt_styles': ['aaa'], 
 'steps': 30, 
 'sampler_index': 16,
  'restore_faces': True,
  'tiling': False, 
  'n_iter': 1, 
  'batch_size': 1, 
  'cfg_scale': 7,
   'seed': 2577302430.0, 
   'subseed': -1.0, 
   'subseed_strength': 0,
    'seed_resize_from_h': 0, 
 'seed_resize_from_w': 0,
 'seed_enable_extras': False, 
 'height': 512, 'width': 512, 
 'enable_hr': False, 'denoising_strength': 0.7, 'hr_scale': 2,
  'hr_upscaler': 'Latent', 'hr_second_pass_steps': 0, 'hr_resize_x': 0, 'hr_resize_y': 0, 'override_settings_texts': [], 'args': 0, 
  'override_settings': <scripts.external_code.ControlNetUnit object at 0x7fab88b5f4d0>, 'p': False, 'processed': False, 'generation_info_js': 'positive'}
"""


# (["<PIL.Image.Image image mode=RGB size=512x512 at 0x7FAB88B4DC10>, <PIL.Image.Image image mode=RGB size=512x512 at 0x7FAB8B9EBCD0>"],
# '{"prompt": "realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),",
# "all_prompts": ["realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),"],
# "negative_prompt": "EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,, EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,",
# "all_negative_prompts": ["EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,, EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,"], "seed": 2577302430,
# "all_seeds": [2577302430], "subseed": 232244428, "all_subseeds": [232244428], "subseed_strength": 0, "width": 512, "height": 512, "sampler_name": "DPM++ SDE Karras", "cfg_scale": 7, "steps": 30, "batch_size": 1, "restore_faces": true, "face_restoration_model": "CodeFormer", "sd_model_hash": "ffe37fe4f3", "seed_resize_from_w": 0, "seed_resize_from_h": 0, "denoising_strength": null, "extra_generation_params": {"ControlNet Enabled": true, "ControlNet Module": "canny", "ControlNet Model": "control_canny-fp16 [e3fe7712]", "ControlNet Weight": 1, "ControlNet Guidance Start": 0, "ControlNet Guidance End": 0.24}, "index_of_first_image": 0,
# "infotexts": ["realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),\\nNegative prompt: EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,, EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,\\nSteps: 30, Sampler: DPM++ SDE Karras, CFG scale: 7, Seed: 2577302430, Face restoration: CodeFormer, Size: 512x512, Model hash: ffe37fe4f3, Model: new, ControlNet Enabled: True, ControlNet Module: canny, ControlNet Model: control_canny-fp16 [e3fe7712], ControlNet Weight: 1, ControlNet Guidance Start: 0, ControlNet Guidance End: 0.24"],
# "styles": ["aaa"], "job_timestamp": "20230417083237", "clip_skip": 1, "is_using_inpainting_conditioning": false}', '<p>realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),, realistic_portrait_female, (white background:1.5), (high detail skin:1.2),<br>\nNegative prompt: EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,, EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,<br>\nSteps: 30, Sampler: DPM++ SDE Karras, CFG scale: 7, Seed: 2577302430, Face restoration: CodeFormer, Size: 512x512, Model hash: ffe37fe4f3, Model: new, ControlNet Enabled: True, ControlNet Module: canny, ControlNet Model: control_canny-fp16 [e3fe7712], ControlNet Weight: 1, ControlNet Guidance Start: 0, ControlNet Guidance End: 0.24</p>', '<p></p>')


def txt2img(
    id_task: str,
    prompt: str,
    negative_prompt: str,
    prompt_styles,
    steps: int,
    sampler_index: int,
    restore_faces: bool,
    tiling: bool,
    n_iter: int,
    batch_size: int,
    cfg_scale: float,
    seed: int,
    subseed: int,
    subseed_strength: float,
    seed_resize_from_h: int,
    seed_resize_from_w: int,
    seed_enable_extras: bool,
    height: int,
    width: int,
    enable_hr: bool,
    denoising_strength: float,
    hr_scale: float,
    hr_upscaler: str,
    hr_second_pass_steps: int,
    hr_resize_x: int,
    hr_resize_y: int,
    override_settings_texts,
    *args,
):
    override_settings = create_override_settings_dict(override_settings_texts)
"""
'outpath_samples': 'outputs/txt2img-images', 'outpath_grids': 'outputs/txt2img-grids', 'prompt': 'realistic_portrait_female, (white background:1.5), black hair, double ponytails, (ponytail front of clothes:1.5)', 'styles': [], 'negative_prompt': 'EasyNegative, (deformed pupils, deformed eyes, dismembered face, 3d, sketch, cartoon, anime:1.4), (light and shadow:1.2), worst quality, out of frame, morbid, (pale skin:1.3), (bangs:1.3), (beard), mutilated, (hands, arms,legs), extra limbs, long neck, signature, watermark, name,', 'seed': 782874815.0, 'subseed': -1.0, 'subseed_strength': 0, 'seed_resize_from_h': 0, 'seed_resize_from_w': 0, 'seed_enable_extras': False, 'sampler_name': 'DPM++ SDE Karras', 'batch_size': 1, 'n_iter': 1, 'steps': 30, 'cfg_scale': 7, 'width': 512, 'height': 512, 'restore_faces': True, 'tiling': False, 'enable_hr': False, 'denoising_strength': None, 'hr_scale': 2, 'hr_upscaler': 'Latent', 'hr_second_pass_steps': 0, 'hr_resize_x': 0, 'hr_resize_y': 0, 'override_settings': {}}


"""
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,  # print model shape here
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples, #outputs/txt2img-images
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers[sampler_index].name,  # DPM++ SDE Karras
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale, #2
        hr_upscaler=hr_upscaler, 
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

"""
{'self': <modules.scripts.ScriptRunner object at 0x7f84557a1390>,
 'p': <modules.processing.StableDiffusionProcessingTxt2Img object at 0x7f8444bbc790>, 
 'args': 0, 
 'script_index': <scripts.external_code.ControlNetUnit object at 0x7f844492cdd0>,
  'script': False, 
  'script_args': False,
   'processed': 'positive'}            
"""

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = process_images(p)

    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return (
        processed.images,
        generation_info_js,
        # plaintext_to_html(processed.info),
        # plaintext_to_html(processed.comments),
    )
