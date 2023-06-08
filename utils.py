from PIL import Image

def make_sheet(img: Image.Image, color: int=255) -> Image.Image:
  res = Image.new('RGBA', (1024,512),color=(color,color,color,color))
  w,h = img.size
  h_offset = int((512-h)/2)
  w_offset = int((256-w)/2)
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