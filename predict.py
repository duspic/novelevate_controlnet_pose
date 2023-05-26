# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
import os
from PIL import Image
from typing import List
import numpy as np

import utils

class Predictor(BasePredictor):
    def setup(self):
       self.model = Model()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        controlnet_pose_image: Path = Input(description="Openpose image with pose to generate"),
        prompt: str = Input(description="Prompt for the model"),
        a_prompt: str = Input(description="Additional text to be appended to prompt", default="""character turnaround sheet, multiple views of the same character,children's book, kid's book, illustration, beautiful, cute,character design, innocent"""),
        n_prompt: str = Input(description="Negative Prompt", default="""NSFW, sexy, seductive, miniskirt, adult,worst quality, low quality, jpeg artifacts, ugly, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers,long neck"""),
        num_images: int = Input(description="Number of samples (higher values may OOM)",
            choices=['1', '2', '3', '4'],
            default='1'
        ),
        image_resolution: str = Input(
            description="Image resolution to be generated",
            choices = ['256', '512', '768'],
            default='512'
        ),
        num_steps: int = Input(description="Steps", default=20),
        cfg_scale: float = Input(description="Scale for classifier-free guidance", default=9.0, ge=0.1, le=30.0),
        seed: int = Input(description="Seed", default=-1),
        strength: float = Input(description="How much noise between 0.0 and 1.0", default=0.8),
        controlnet_strength: float = Input(description="How much to follow controlnet between 0.0 and 2.0", default=1.9)
        

    ) -> List[Path]:
        """Run a single prediction on the model"""
        
        input_img = Image.open(image)
        input_img = utils.scale_for_sheet(input_img)
        input_img = utils.make_sheet(input_img)
        input_img = np.array(input_img)
        
        pose_img = Image.open(controlnet_pose_image)
        pose_img = utils.scale_for_sheet(pose_img)
        pose_img = utils.make_sheet(pose_img, 0)
        pose_img = np.array(pose_img)  
        
        mask_img = utils.make_mask()
        mask_img = np.array(mask_img)

        if not input_img.shape == pose_img.shape == mask_img.shape:
            raise ValueError(f"""The mask, pose and input image must have the same shape
                             input_img{input_img.shape}, pose_img{pose_img.shape}, mask_img{mask_img.shape}
                             """)
        
        outputs = self.model.process_openpose(
            image=input_img,
            mask_image=mask_img,
            controlnet_conditioning_image=pose_img,
            prompt=prompt,
            additional_prompt=a_prompt,
            negative_prompt=n_prompt,
            num_images=num_images,
            image_resolution=image_resolution,
            num_steps=num_steps,
            seed=seed,
            strength=strength,
            controlnet_conditioning_scale=controlnet_strength,
            guidance_scale=cfg_scale
        )

        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        
        outputs = [utils.extract_char(output) for output in outputs]
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        return [Path(f"./tmp/output_{i}.png") for i in range(len(outputs))]
