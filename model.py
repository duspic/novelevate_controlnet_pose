from __future__ import annotations

import gc

import numpy as np
import PIL.Image
import torch
import cv2
import einops
from transformers import AutoModel
from annotator.util import resize_image, HWC3
from diffusers import (ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler,
                       EulerAncestralDiscreteScheduler)

import textual_inversion_fix # hopefully

CONTROLNET_MODEL_IDS = {
    'Openpose': 'lllyasviel/control_v11p_sd15_openpose',
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'MLSD': 'lllyasviel/control_v11p_sd15_mlsd',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'softedge': 'lllyasviel/control_v11p_sd15_softedge',
    'segmentation': 'lllyasviel/control_v11p_sd15_seg',
    'depth': 'lllyasviel/control_v11f1p_sd15_depth',
    'NormalBae': 'lllyasviel/control_v11p_sd15_normalbae',
    'lineart': 'lllyasviel/control_v11p_sd15_lineart',
    'lineart_anime': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
    'shuffle': 'lllyasviel/control_v11e_sd15_shuffle',
    'ip2p': 'lllyasviel/control_v11e_sd15_ip2p',
    'inpaint': 'lllyasviel/control_v11e_sd15_inpaint',
}
# TODO different model
config_dict = {
    'Uminosachi/dreamshaper_5-inpainting': 'lllyasviel/control_v11p_sd15_openpose',
}


class Model:
    def __init__(self,
                 base_model_id: str = 'Uminosachi/dreamshaper_5-inpainting',
                 task_name: str = 'Openpose'):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.base_model_id = ''
        self.task_name = ''
        self.pipe = self.load_pipe(base_model_id, task_name)

    def load_pipe(self, base_model_id: str, task_name) -> DiffusionPipeline:
        if base_model_id == self.base_model_id and task_name == self.task_name and hasattr(
                self, 'pipe') and self.pipe is not None:
            return self.pipe

        controlnet = ControlNetModel.from_pretrained(config_dict[base_model_id])
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
            custom_pipeline="stable_diffusion_controlnet_inpaint_img2img")
        
        textual_inversion_fix.load_textual_inversion(pipe, "charturnerv2.pt", token="charturnerv2")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        #if self.device.type == 'cuda':
        #    pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, 'controlnet'):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=torch.float16)
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f'{prompt}, {additional_prompt}'
        return prompt

    @torch.autocast('cuda')
    def run_pipe(
        self,
        prompt: str,
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        controlnet_conditioning_image: PIL.Image.Image,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        num_images_per_prompt: int,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: float=1.9,
        ) -> list[PIL.Image.Image]:
        
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int64).max)
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            controlnet_conditioning_image=controlnet_conditioning_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            strength=strength
            ).images

    @torch.inference_mode()
    def process_openpose(
        self,
        image: np.ndarray,
        mask_image: np.ndarray,
        controlnet_conditioning_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: float=1.9,
    ) -> list[PIL.Image.Image]:
        img = resize_image(HWC3(image), image_resolution)
        control = resize_image(HWC3(controlnet_conditioning_image), image_resolution)
        mask = resize_image(HWC3(mask_image), image_resolution)

        H, W, _ = img.shape
        
        img = PIL.Image.fromarray(img)
        control = PIL.Image.fromarray(control)
        mask = PIL.Image.fromarray(mask)
        
        self.load_controlnet_weight('Openpose')
        results = self.run_pipe(
            prompt=f"charturnerv2, {prompt}, {additional_prompt}",
            image=img,
            mask_image=mask,
            controlnet_conditioning_image=control,
            height=H,
            width=W,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=seed,
            strength=strength
        )
        return results

    def openpose(self, img, has_hand=False):
        from annotator.openpose import apply_openpose
        result, _ = apply_openpose(img, has_hand)
        return [result]