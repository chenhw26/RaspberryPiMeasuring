from PIL import Image

def load_img(path, target_size=None, crop_size=None):
  img = Image.open(path)

  if crop_size:
    half_the_width = img.size[0] / 2
    img = img.crop((half_the_width - int(crop_size[0] / 2),
                    img.size[1] - crop_size[1],
                    half_the_width + int(crop_size[0] / 2),
                    img.size[1]))

  if target_size:
    img = img.resize(target_size)

  return img