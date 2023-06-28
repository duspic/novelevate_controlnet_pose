from __future__ import annotations

import gc

import numpy as np
import PIL.Image
import torch

from transformers import AutoModel
from annotator.util import resize_image, HWC3
from diffusers import (ControlNetModel, StableDiffusionControlNetInpaintPipeline,
                       EulerAncestralDiscreteScheduler, AutoencoderKL)

from typing import Union, List
from utils import make_inpaint_condition


class Model:
    def __init__(self,
                 base_model_id: str = 'ducnapa/cute-cartoon-illustration',
                 ):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.base_model_id = ''
        self.pipe = self.load_pipe(base_model_id)

    def load_pipe(self, base_model_id: str) -> StableDiffusionControlNetInpaintPipeline:
        if base_model_id == self.base_model_id and hasattr(
                self, 'pipe') and self.pipe is not None:
            return self.pipe

        controlnet = [
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)
            ]
        
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "ducnapa/cute-cartoon-illustration",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            vae = AutoencoderKL.from_pretrained('bullhug/blessed_vae')
            )
        
        pipe.load_textual_inversion("charturnerv2.pt", token="charturnerv2")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        #if self.device.type == 'cuda':
        #    pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id)
        return self.base_model_id

    def load_controlnet_weight(self) -> None:
        if self.pipe.controlnet:
            return
        if self.pipe is not None and hasattr(self.pipe, 'controlnet'):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        
        controlnet = [
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16)
        ]
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet

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
        control_image: List[Union[PIL.Image.Image, torch.Tensor]],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        negative_prompt: str,
        num_images_per_prompt: int,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: list[float],
        ) -> list[PIL.Image.Image]:
        
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int64).max)
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            control_image=control_image,
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
        control_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: List[float],
    ) -> list[PIL.Image.Image]:
        img = resize_image(HWC3(image), image_resolution)
        pose = resize_image(HWC3(control_image), image_resolution)
        mask = resize_image(HWC3(mask_image), image_resolution)

        H, W, _ = img.shape
        
        img = PIL.Image.fromarray(img)
        pose = PIL.Image.fromarray(pose)
        mask = PIL.Image.fromarray(mask)
        inpaint_img = make_inpaint_condition(img, mask)
        
        self.load_controlnet_weight()
        results = self.run_pipe(
            prompt=f"charturnerv2, {prompt}, {additional_prompt}",
            image=img,
            mask_image=mask,
            control_image=[pose, inpaint_img],
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