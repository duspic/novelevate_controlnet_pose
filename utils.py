from PIL import Image, ImageOps

def make_pose_sheet(img: Image.Image, color: int=255, len=4) -> Image.Image:
  res = Image.new('RGBA', (len*256,512),color=(color,color,color,color))
  w,h = img.size
  h_offset = int((512-h)/2)
  w_offset = int((256-w)/2)
  for i in range(len):
    res.paste(img, mask=img, box=(i*256 + w_offset,h_offset))

  return res.convert('RGB')

def make_mask(len=4) -> Image.Image:
  res = Image.new('RGB', (len*256,512), color=(0,0,0))
  white = Image.new('RGB', (256,512), color=(255,255,255))

  res.paste(white, box=(512,0))
  return res


def scale_for_sheet(img: Image.Image, invert=False) -> Image.Image:
  maxwidth = 256 
  w,h = img.size
  ratio = maxwidth/w
  return img.resize((256,int(ratio*h)))


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


def make_character_sheet(img: Image.Image, len=4) -> Image.Image:
    res = Image.new('RGBA', (len*256,512),color=(255,255,255,255))
    w,h = img.size
    
    # first scramble
    img2 = img.resize((w*2,h*2))
    img2 = img2.crop((3*w//4,0,7*w//4,h))
    w2,h2 = img2.size
    h_offset = int((512-h2)/2)
    w_offset = int((256-w2)/2)
    res.paste(img2, mask=img2, box=(w_offset,h_offset))

    # paste regular image at second place
    
    h_offset = int((512-h)/2)
    w_offset = int((256-w)/2)
    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    res.paste(img2, mask=img2, box=(256+w_offset,h_offset))

    """# second scramble
    img2 = img.resize((w*3, h*3))
    w2,h2 = img2.size
    h_offset = int((512-h2)/2)
    w_offset = int((256-w2)/2)
    img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    res.paste(img2, mask=img2, box=(256+w_offset,h_offset))"""

    # leave room at third place
    # paste regular image at fourth
    
    h_offset = int((512-h)/2)
    w_offset = int((256-w)/2)
    res.paste(img, mask=img, box=(768+w_offset,h_offset))
    
    return res.convert('RGB')