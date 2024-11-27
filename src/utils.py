import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image

def load_pipeline(args, device=None) -> StableDiffusionPipeline:
    scheduler = DDIMScheduler.from_config(args.model_base, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_base,
        safety_checker=None,
        scheduler=scheduler,
    )
    if device is not None:
        pipe.to(device)
    return pipe
