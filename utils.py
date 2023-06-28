from PIL import Image
import numpy as np
import torch

def make_sheet(img: Image.Image, color: int=255, onlythird: bool=False) -> Image.Image:
  res = Image.new('RGBA', (1024,512),color=(color,color,color,color))
  w,h = img.size
  h_offset = int((512-h)/2)
  w_offset = int((256-w)/2)

  if onlythird:
    res.paste(img, mask=img, box=(512+w_offset, h_offset))
    return res.convert('RGB')

  for i in range(4):
    res.paste(img, mask=img, box=(i*256 + w_offset,h_offset))
  return res.convert('RGB')

def make_mask() -> Image.Image:
  res = Image.new('RGB', (1024,512), color=(0,0,0))
  white = Image.new('RGB', (256,512), color=(255,255,255))

  res.paste(white, box=(512,0))
  return res


def scale_for_sheet(img: Image.Image) -> Image.Image:
  maxwidth = 256 
  w,h = img.size
  ratio = maxwidth/w
  return img.resize((256,int(ratio*h)))


def extract_char(res: Image.Image) -> Image.Image:
  return res.crop((512,0,768,512))

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image