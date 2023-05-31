from PIL import Image, ImageOps

def make_sheet(img: Image.Image, color: int=255, skip_third=False, len=4) -> Image.Image:
  res = Image.new('RGBA', (len*256,512),color=(color,color,color,color))
  w,h = img.size
  h_offset = int((512-h)/2)
  w_offset = int((256-w)/2)
  for i in range(len):
    if i==2 and skip_third:
      continue
    res.paste(img, mask=img, box=(i*256 + w_offset,h_offset))

  return res.convert('RGB')

def make_mask(len=4) -> Image.Image:
  res = Image.new('RGB', (len*256,512), color=(0,0,0))
  white = Image.new('RGB', (256,512), color=(255,255,255))

  res.paste(white, box=(512,0))
  return res


def scale_for_sheet(img: Image.Image, invert=False) -> Image.Image:
  maxwidth = 256
  
  if invert:
    im_crop = img.crop(invert_color(img).getbbox())
  else:
    im_crop = img.crop(img.getbbox())
 
  w,h = im_crop.size
  if w > maxwidth:
    ratio = maxwidth/w
    h = int(ratio*h)
    return im_crop.resize((256,h))

  return im_crop


def extract_char(res: Image.Image) -> Image.Image:
  return res.crop((512,0,768,512))


def invert_color(img: Image.Image) -> Image:
  if img.mode == "RGBA":
    r,g,b,a = img.split()
    rgb_image = Image.merge('RGB', (r,g,b))
    inverted_image = ImageOps.invert(rgb_image)
    r2,g2,b2 = inverted_image.split()
    return Image.merge('RGBA', (r2,g2,b2,a))
  
  return ImageOps.invert(img)