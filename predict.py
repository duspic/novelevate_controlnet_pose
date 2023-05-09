# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
from annotator.util import resize_image, HWC3
import cv2
import os
from PIL import Image
from typing import List
import numpy as np

SD_15="runwayml/stable-diffusion-v1-5"
PROTOGEN="darkstorm2150/Protogen_x5.8_Official_Release"
MODELSHOOT="wavymulder/modelshoot"
SD_21="stabilityai/stable-diffusion-2-1-base"
ANIME="DGSpitzer/Cyberpunk-Anime-Diffusion"

class Predictor(BasePredictor):
    def setup(self):
       self.model = Model(base_model_id=SD_21, task_name='canny')

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt for the model"),
        num_samples: str = Input(
            description="Number of samples (higher values may OOM)",
            choices=['1', '4'],
            default='1'
        ),
        image_resolution: str = Input(
            description="Image resolution to be generated",
            choices = ['256', '512', '768'],
            default='512'
        ),
        low_threshold: int = Input(description="Canny line detection low threshold", default=100, ge=1, le=255), # only applicable when model type is 'canny'
        high_threshold: int = Input(description="Canny line detection high threshold", default=200, ge=1, le=255), # only applicable when model type is 'canny'
        ddim_steps: int = Input(description="Steps", default=20),
        scale: float = Input(description="Scale for classifier-free guidance", default=9.0, ge=0.1, le=30.0),
        seed: int = Input(description="Seed", default=-1),
        eta: float = Input(description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise", default=0.0),
        a_prompt: str = Input(description="Additional text to be appended to prompt", default="best quality, extremely detailed"),
        n_prompt: str = Input(description="Negative Prompt", default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        detect_resolution: int = Input(description="Resolution at which detection method will be applied)", default=512, ge=128, le=1024), # only applicable when model type is 'HED', 'seg', or 'MLSD'
        # bg_threshold: float = Input(description="Background Threshold (only applicable when model type is 'normal')", default=0.0, ge=0.0, le=1.0), # only applicable when model type is 'normal'
        # value_threshold: float = Input(description="Value Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=2.0), # only applicable when model type is 'MLSD'
        # distance_threshold: float = Input(description="Distance Threshold (only applicable when model type is 'MLSD')", default=0.1, ge=0.01, le=20.0), # only applicable when model type is 'MLSD'
    ) -> List[Path]:
        """Run a single prediction on the model"""

        input_image = Image.open(image)
        input_image = np.array(input_image)        

        outputs = self.model.process_canny(
            input_image,
            prompt,
            a_prompt,
            n_prompt,
            int(num_samples),
            image_resolution,
            ddim_steps,
            scale,
            seed,
            low_threshold,
            high_threshold
        )


        # outputs = [Image.fromarray(output) for output in outputs]

        all_files = os.listdir("tmp/")
        existing_images = [filename for filename in all_files if filename.startswith("output_") and filename.endswith(".png")]
        num_existing_images = len(existing_images)

        outputs = [output.save(f"tmp/output_{num_existing_images+i}.png") for i, output in enumerate(outputs)]
        return [Path(f"tmp/output_{num_existing_images+i}.png") for i in range(len(outputs))]
