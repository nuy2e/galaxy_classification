"""
Image Generator using Stable Diffusion and LoRA Adapters

This script generates a specified number of images using the Stable Diffusion v1.5 model
augmented with a LoRA adapter. Images are generated based on a user-defined text prompt
and saved to a target output directory.

Features:
- Supports configurable image resolution, prompt, and guidance parameters.
- Loads custom LoRA weights for fine-tuned generation.
- Uses Euler Ancestral scheduler for image sampling.
- Optionally supports deterministic output via seed (commented by default).
- Saves images with sequentially numbered filenames and prints progress every 100 images.

Dependencies:
- diffusers
- torch
- Pillow (PIL)

Usage:
- Set `USE_DEFAULT_PATHS` to configure internal/external paths.
- Modify `prompt`, `image_*`, `num_images`, etc. for custom generation.
- Run the script directly with Python to start generation.

Note:
To enable deterministic image generation, uncomment and configure the `seed_base` and `generator`.
"""

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
from PIL import Image
import random

# === Config ===
USE_DEFAULT_PATHS = True

if USE_DEFAULT_PATHS:
    lora_path = os.path.join("lora_model", "E.safetensors")
    output_dir = os.path.join("generated_data")
else:
    lora_path = r"path\to\lora"
    output_dir = r"path\to\output"

base_model = r"runwayml/stable-diffusion-v1-5"
prompt = "elliptical galaxy"
negative_prompt = ""

#Image setup
image_width = 256
image_height = 256
num_images = 4350
num_inference_steps = 20
guidance_scale = 7
batch_size = 1  # use >1 if you want to batch for speed

#uncomment for psudo-random images
#seed_base = 42  # change this for variation

# === Setup ===
os.makedirs(output_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

#Scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load LoRA
pipe.load_lora_weights(lora_path)
pipe.set_adapters(["default_0"], [1])  # or [0.0] to disable
# Disable progress bar
pipe.set_progress_bar_config(disable=True)

# === Generate Images ===
for i in range(num_images):

    #uncomment this to generate psudo-random images
    #generator = torch.Generator("cuda").manual_seed(seed_base + i)
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        #generator=generator
    )

    image: Image.Image = result.images[0]
    image.save(os.path.join(output_dir, f"galaxy_{i:05}.png"))

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}/{num_images}] images generated")
