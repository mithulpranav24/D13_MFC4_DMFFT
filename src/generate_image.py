#!/usr/bin/env python3
"""
Simple image generation script using Hugging Face Diffusers.
Generates a single image from a text prompt and saves it as a PNG.

Usage:
    python generate_image.py --prompt "A cute cat" --steps 50 --seed 42

Notes:
- This script intentionally contains NO FFT/DMFFT code — it only runs the diffusion model.
- If you need a gated StabilityAI model, set environment variable HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`.
"""
import argparse
import os
import sys

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
except Exception as e:
    print("Missing dependencies. Install requirements: pip install diffusers[torch] transformers accelerate safetensors pillow")
    raise


def load_pipeline(model_id, hf_token=None, device_str="auto"):
    # choose device
    if device_str == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    torch_dtype = torch.float16 if (device == "cuda") else torch.float32

    try:
        if hf_token:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, use_auth_token=hf_token)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

    pipe = pipe.to(device)
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help='Hugging Face model id')
    parser.add_argument('--out', type=str, default='generated.png')
    parser.add_argument('--device', type=str, default='auto', help='cuda|cpu|auto')
    args = parser.parse_args()

    hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')

    # Try primary model, fallback to a public model if gated/unavailable
    fallback = 'runwayml/stable-diffusion-v1-5'
    try:
        pipe = load_pipeline(args.model, hf_token=hf_token, device_str=args.device)
    except Exception as e:
        print(f"Warning: could not load primary model {args.model}: {e}")
        print(f"Attempting fallback model {fallback}...")
        try:
            pipe = load_pipeline(fallback, hf_token=None, device_str=args.device)
        except Exception as e2:
            print(f"Failed to load fallback model: {e2}")
            sys.exit(1)

    # Set seed
    generator = None
    if args.seed is not None:
        device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed))
        generator = gen

    # Generate
    print(f"Generating with model={pipe.__class__.__name__}, device={pipe.device}, steps={args.steps}")
    result = pipe(args.prompt, num_inference_steps=int(args.steps), generator=generator)
    image = result.images[0]

    # Save
    out_path = os.path.abspath(args.out)
    image.save(out_path)
    print(f"Saved image to {out_path}")


if __name__ == '__main__':
    main()
