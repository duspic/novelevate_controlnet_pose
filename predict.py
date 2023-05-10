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

REALISTICVISION = "SG161222/Realistic_Vision_V2.0"

class Predictor(BasePredictor):
    def setup(self):
       self.model = Model(base_model_id=REALISTICVISION, task_name='canny')

    def predict(
        self,
        image: Path = Input(description="Background-removed image"),
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
        a_prompt: str = Input(description="Additional text to be appended to prompt", default="RAW photo, product photography, highres, extremely detailed, best quality,  8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3,"),
        n_prompt: str = Input(description="Negative Prompt", default="poorly drawn, lowres, bad quality, worst quality, unrealistic, overexposed, underexposed, floating, blurry background"),
        detect_resolution: int = Input(description="Resolution at which detection method will be applied)", default=512, ge=128, le=1024), # only applicable when model type is 'HED', 'seg', or 'MLSD'
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


        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        
        noback_img = Image.open(image)
        for output in outputs:
            output.paste(noback_img, mask=noback_img)
        outputs = [output.save(f"tmp/output_{i}.png") for i, output in enumerate(outputs)]
        return [Path(f"./tmp/output_{i}.png") for i in range(len(outputs))]
