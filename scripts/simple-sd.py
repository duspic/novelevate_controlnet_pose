from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "DGSpitzer/Cyberpunk-Anime-Diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a beautiful perfect face girl in dgs illustration style, Anime fine details portrait of school girl in front of modern tokyo city landscape on the background deep bokeh, anime masterpiece, 8k, sharp high quality anime"
image = pipe(prompt).images[0]

image.save("./cyberpunk_girl.png")