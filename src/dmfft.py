import os
import re
import textwrap
from datetime import datetime

import torch
import torch.fft as fft
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_torch_version
from PIL import Image, ImageDraw, ImageFont

def isinstance_str(x, cls_name):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False

def fourier_filter(x, threshold, scale, verbose=False):
    if threshold <= 0:
        return x

    dtype = x.dtype
    x = x.type(torch.float32)

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)

    crow, ccol = H // 2, W // 2
    top    = max(0, crow - threshold)
    bottom = min(H, crow + threshold)
    left   = max(0, ccol - threshold)
    right  = min(W, ccol + threshold)

    if verbose:
        lf_mean = torch.abs(x_freq[..., top:bottom, left:right]).mean().item()
        hf_mask = torch.ones_like(x_freq, dtype=torch.bool)
        hf_mask[..., top:bottom, left:right] = False
        hf_mean = torch.abs(x_freq[hf_mask]).mean().item()
        print("  [fourier_filter] LF mean={:.4f}  HF mean={:.4f}  "
              "region=[{}:{}, {}:{}]  scale={}".format(
                  lf_mean, hf_mean, top, bottom, left, right, scale))

    mask[..., top:bottom, left:right] = scale
    x_freq = x_freq * mask

    x_freq     = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered.type(dtype)

def fourier_solo(x, global_scale=1.0, freq_threshold=1,
                 lf_scale=1.0, hf_scale=1.0,
                 amplitude_scale=1.0, phase_scale=1.0,
                 blend_type=0, verbose=False):
    dtype = x.dtype
    x = x.type(torch.float32)
    x = x * global_scale

    if blend_type == -1 or freq_threshold <= 0:
        return x.type(dtype)

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape

    crow, ccol = H // 2, W // 2
    top    = max(0, crow - freq_threshold)
    bottom = min(H, crow + freq_threshold)
    left   = max(0, ccol - freq_threshold)
    right  = min(W, ccol + freq_threshold)

    amplitude = torch.abs(x_freq)
    phase     = torch.angle(x_freq)

    if verbose:
        lf_amp_mean = amplitude[..., top:bottom, left:right].mean().item()
        hf_mask     = torch.ones_like(amplitude, dtype=torch.bool)
        hf_mask[..., top:bottom, left:right] = False
        hf_amp_mean = amplitude[hf_mask].mean().item()
        print("  [fourier_solo]   LF amp={:.4f}  HF amp={:.4f}  "
              "region=[{}:{}, {}:{}]  blend={}".format(
                  lf_amp_mean, hf_amp_mean, top, bottom, left, right, blend_type))

    if blend_type == 0:
        amplitude = amplitude * amplitude_scale
        phase     = phase * phase_scale

    elif blend_type == 1:
        amplitude[..., top:bottom, left:right] = (
            amplitude[..., top:bottom, left:right] * amplitude_scale)
        phase[..., top:bottom, left:right] = (
            phase[..., top:bottom, left:right] * phase_scale)

    elif blend_type == 2:
        lf_amp   = amplitude[..., top:bottom, left:right].clone()
        lf_phase = phase[..., top:bottom, left:right].clone()
        amplitude = amplitude * amplitude_scale
        phase     = phase * phase_scale
        amplitude[..., top:bottom, left:right] = lf_amp
        phase[..., top:bottom, left:right]     = lf_phase

    elif blend_type == 3:
        amplitude[..., top:bottom, left:right] = (
            amplitude[..., top:bottom, left:right] * amplitude_scale)
        lf_phase = phase[..., top:bottom, left:right].clone()
        phase    = phase * phase_scale
        phase[..., top:bottom, left:right] = lf_phase

    elif blend_type == 4:
        lf_amp = amplitude[..., top:bottom, left:right].clone()
        amplitude = amplitude * amplitude_scale
        amplitude[..., top:bottom, left:right] = lf_amp
        phase[..., top:bottom, left:right] = (
            phase[..., top:bottom, left:right] * phase_scale)

    if lf_scale != 1.0 or hf_scale != 1.0:
        freq_mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)
        freq_mask[...] = hf_scale
        freq_mask[..., top:bottom, left:right] = lf_scale
        amplitude = amplitude * freq_mask

    reals      = amplitude * torch.cos(phase)
    imags      = amplitude * torch.sin(phase)
    x_freq_new = torch.complex(reals, imags)

    x_freq_new = fft.ifftshift(x_freq_new, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq_new, dim=(-2, -1)).real
    return x_filtered.type(dtype)

def _set_attrs(block, **kwargs):
    for k, v in kwargs.items():
        setattr(block, k, v)

def register_tune_crossattn_upblock2d(model, use_fft=True, verbose=False,
                                      types=0,
                                      k2=0.5,   b2=1.4,  t2=1,   s2=0.2,
                                      k2_1=0.5, b2_1=1.4, t2_1=1, s2_1=0.2,
                                      g2=1.0,   g2_1=1.0,
                                      blend2=0, blend2_1=0,
                                      a2=1.0,   a2_1=1.0,
                                      p2=1.0,   p2_1=1.0,
                                      skips=0,  tunes=0):
    def up_forward(block):
        def forward(hidden_states, res_hidden_states_tuple, temb=None,
                    encoder_hidden_states=None, cross_attention_kwargs=None,
                    upsample_size=None, attention_mask=None,
                    encoder_attention_mask=None):

            skipping = 0
            finetune = 0

            for resnet, attn in zip(block.resnets, block.attentions):
                res_hidden_states       = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if use_fft:
                    C = hidden_states.shape[1]

                    if block.types == 1:
                        hidden_states[:, :int(C * block.k2)] *= block.b2
                        res_hidden_states = fourier_filter(
                            res_hidden_states,
                            threshold=block.t2, scale=block.s2,
                            verbose=verbose)

                    elif block.types == 2:
                        tunes_limit = block.tunes if block.tunes > 0 else float('inf')
                        if skipping >= block.skips and finetune < tunes_limit:
                            hidden_states[:, :int(C * block.k2)] *= block.b2
                            res_hidden_states[:, :int(
                                res_hidden_states.shape[1] * block.k2_1)] *= block.s2
                            finetune += 1

                    elif block.types == 3:
                        import numpy as _np
                        hs_np  = hidden_states.to('cpu').detach().float().numpy()
                        avg    = _np.mean(hs_np, axis=1)
                        hs_max = _np.max(avg, axis=(1, 2))
                        hs_min = _np.min(avg, axis=(1, 2))
                        n = int(C * block.k2)
                        for idx in range(hidden_states.shape[0]):
                            span = max(hs_max[idx] - hs_min[idx], 1e-6)
                            hidden_states[idx, :n] *= (block.b2 * span)
                        res_hidden_states = fourier_filter(
                            res_hidden_states,
                            threshold=block.t2, scale=block.s2,
                            verbose=verbose)

                    elif block.types == 4:
                        tunes_limit = block.tunes if block.tunes > 0 else float('inf')
                        if skipping >= block.skips and finetune < tunes_limit:
                            hidden_states = fourier_solo(
                                hidden_states,
                                global_scale=block.g2, freq_threshold=block.t2,
                                lf_scale=block.s2,     hf_scale=block.b2,
                                amplitude_scale=block.a2, phase_scale=block.p2,
                                blend_type=block.blend2, verbose=verbose)
                            res_hidden_states = fourier_solo(
                                res_hidden_states,
                                global_scale=block.g2_1, freq_threshold=block.t2_1,
                                lf_scale=block.s2_1,     hf_scale=block.b2_1,
                                amplitude_scale=block.a2_1, phase_scale=block.p2_1,
                                blend_type=block.blend2_1, verbose=verbose)
                            finetune += 1

                    elif block.types == 5:
                        hidden_states = fourier_solo(
                            hidden_states,
                            global_scale=block.g2, freq_threshold=block.t2,
                            lf_scale=block.s2,     hf_scale=block.b2,
                            amplitude_scale=block.a2, phase_scale=block.p2,
                            blend_type=block.blend2, verbose=verbose)
                        res_hidden_states = fourier_solo(
                            res_hidden_states,
                            global_scale=block.g2_1, freq_threshold=block.t2_1,
                            lf_scale=block.s2_1,     hf_scale=block.b2_1,
                            amplitude_scale=block.a2_1, phase_scale=block.p2_1,
                            blend_type=block.blend2_1, verbose=verbose)

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                use_gc = (block.training and
                          getattr(block, "gradient_checkpointing", False))

                if use_gc:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            return module(*inputs)
                        return custom_forward

                    ckpt_kwargs = ({"use_reentrant": False}
                                   if is_torch_version(">=", "1.11.0") else {})
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states, temb, **ckpt_kwargs)
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states, encoder_hidden_states, None, None,
                        cross_attention_kwargs, attention_mask,
                        encoder_attention_mask, **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )[0]

                skipping += 1

            if block.upsamplers is not None:
                for upsampler in block.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    crossattn_count = 0
    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            crossattn_count += 1
            if crossattn_count == 3:
                _set_attrs(
                    upsample_block,
                    types=types,
                    k2=k2,       b2=b2,       t2=t2,       s2=s2,
                    k2_1=k2_1,   b2_1=b2_1,   t2_1=t2_1,   s2_1=s2_1,
                    g2=g2,       g2_1=g2_1,
                    blend2=blend2,     blend2_1=blend2_1,
                    a2=a2,       a2_1=a2_1,
                    p2=p2,       p2_1=p2_1,
                    skips=skips, tunes=tunes,
                )
                upsample_block.forward = up_forward(upsample_block)
                break

def _load_pipeline(model_id, device, dtype, disable_safety_checker=False):
    kwargs = {"torch_dtype": dtype}
    if disable_safety_checker:
        kwargs.update({"safety_checker": None, "feature_extractor": None})

    pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

    if disable_safety_checker:
        pipe.safety_checker          = None
        pipe.requires_safety_checker = False

    return pipe.to(device)

def _load_font(size):
    candidates = [
        "arial.ttf", "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _textsize(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def _prompt_slug(prompt, max_len=40):
    slug = prompt.lower()
    slug = re.sub(r"[^a-z0-9 ]", "", slug)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:max_len]

def _save_comparison(img_baseline, img_dmfft, prompt, out_dir, index):
    img_w, img_h = img_dmfft.size

    if img_baseline.size != img_dmfft.size:
        img_baseline = img_baseline.resize(img_dmfft.size, Image.LANCZOS)

    LABEL_H  = 40
    FOOTER_H = 60
    GAP      = 6

    canvas_w = img_w * 2 + GAP
    canvas_h = LABEL_H + img_h + FOOTER_H

    canvas = Image.new("RGB", (canvas_w, canvas_h), (14, 14, 20))
    draw   = ImageDraw.Draw(canvas)

    font_label  = _load_font(16)
    font_footer = _load_font(13)

    draw.rectangle([0,           0, img_w - 1,   LABEL_H], fill=(36, 36, 48))
    draw.rectangle([img_w + GAP, 0, canvas_w - 1, LABEL_H], fill=(12, 38, 84))

    def _centre_text(text, x0, x1, y0, y1, font, color):
        tw, th = _textsize(draw, text, font)
        draw.text(
            (x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2),
            text, font=font, fill=color)

    _centre_text("Original (Baseline)", 0, img_w, 0, LABEL_H,
                 font_label, (185, 185, 200))
    _centre_text("DMFFT Enhanced", img_w + GAP, canvas_w, 0, LABEL_H,
                 font_label, (80, 180, 255))

    canvas.paste(img_baseline, (0,           LABEL_H))
    canvas.paste(img_dmfft,    (img_w + GAP, LABEL_H))

    draw.rectangle([img_w, 0, img_w + GAP - 1, LABEL_H + img_h],
                   fill=(44, 44, 56))

    footer_y = LABEL_H + img_h
    draw.rectangle([0, footer_y, canvas_w, canvas_h], fill=(10, 10, 14))
    draw.rectangle([0, footer_y, canvas_w, footer_y + 1], fill=(48, 48, 60))

    max_chars = max(40, canvas_w // 9)
    lines     = textwrap.wrap('"{}"'.format(prompt), width=max_chars)
    LINE_H    = 18
    total_h   = len(lines) * LINE_H
    ty        = footer_y + max(4, (FOOTER_H - total_h) // 2)

    for i, line in enumerate(lines):
        tw, _ = _textsize(draw, line, font_footer)
        draw.text(((canvas_w - tw) // 2, ty + i * LINE_H),
                  line, font=font_footer, fill=(150, 150, 168))

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug     = _prompt_slug(prompt)
    filename = os.path.join(
        out_dir, "dmfft_{:03d}_{}_{}.png".format(index, slug, ts))
    canvas.save(filename)
    return filename

def _save_grid(comparison_paths, out_dir):
    images = [Image.open(p) for p in comparison_paths]
    w      = images[0].width
    h_each = images[0].height
    total_h = h_each * len(images)

    grid = Image.new("RGB", (w, total_h), (10, 10, 14))
    for i, img in enumerate(images):
        grid.paste(img, (0, i * h_each))

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, "dmfft_grid_{}.png".format(ts))
    grid.save(filename)
    return filename

def _run_prompt(pipe, prompt, seed, steps, dmfft_params, verbose, out_dir, index):
    print("\n  Prompt {}: {}".format(index, prompt))
    print("  Seed  : {}".format(seed))

    print("  [1/2] Baseline...")
    register_tune_crossattn_upblock2d(
        pipe, use_fft=False, verbose=verbose, **dmfft_params)
    torch.manual_seed(seed)
    img_baseline = pipe(prompt, num_inference_steps=steps).images[0]

    print("  [2/2] DMFFT...")
    register_tune_crossattn_upblock2d(
        pipe, use_fft=True, verbose=verbose, **dmfft_params)
    torch.manual_seed(seed)
    img_dmfft = pipe(prompt, num_inference_steps=steps).images[0]

    out_path = _save_comparison(
        img_baseline, img_dmfft, prompt, out_dir, index)
    print("  [OK]  Saved -> {}".format(out_path))
    return out_path

def main():
    PROMPTS = [
        "a lone lighthouse on a rocky coast during a storm, dramatic lighting",
        "a Japanese tea ceremony in a bamboo garden, soft morning light",
        "a giant sea turtle swimming through a coral reef, underwater photography",
        "a vintage steam train crossing a snow-covered mountain bridge",
        "a street market in Marrakech at dusk, vibrant colors and lanterns",
        "a wolf howling on a frozen tundra under the northern lights",
        "a close-up portrait of an old fisherman with weathered skin and kind eyes",
        "a child reading a glowing book in a dark enchanted forest",
    ]

    SEED    = 880
    STEPS   = 25
    MODEL   = "runwayml/stable-diffusion-v1-5"
    OUT_DIR = "results"

    dmfft_params = dict(
        types=1,
        k2=0.3, b2=1.1, t2=1, s2=0.95,
        k2_1=0.3, b2_1=1.1, t2_1=1, s2_1=0.95,
        g2=1.0, g2_1=1.0,
        blend2=0, blend2_1=0,
        a2=1.0, a2_1=1.0,
        p2=1.0, p2_1=1.0,
        skips=0, tunes=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print("Loading '{}' on {} ({})...".format(MODEL, device, dtype))
    pipe = _load_pipeline(MODEL, device, dtype, disable_safety_checker=False)
    os.makedirs(OUT_DIR, exist_ok=True)

    saved = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print("\n[{}/{}] {}".format(i, len(PROMPTS), prompt))
        path = _run_prompt(
            pipe=pipe, prompt=prompt, seed=SEED, steps=STEPS,
            dmfft_params=dmfft_params, verbose=True,
            out_dir=OUT_DIR, index=i,
        )
        saved.append(path)

    grid_path = _save_grid(saved, OUT_DIR)
    print("\n[GRID] {}".format(grid_path))
    print("Done. All images saved to '{}'.".format(os.path.abspath(OUT_DIR)))

if __name__ == "__main__":
    main()