import torch, numpy as np 
from PIL import Image

def make_batch(image, mask, texture, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    texture_image = np.array(Image.open(texture).convert("RGB"))
    texture_image = texture_image.astype(np.float32)/255.0
    texture_image = texture_image[None].transpose(0,3,1,2)
    texture_image = torch.from_numpy(texture_image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image, "texture":texture_image}

    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch