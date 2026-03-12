import argparse
import os

import numpy
import torch
import torch.fft as fft
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_torch_version


def isinstance_str(x, cls_name):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def fourier_filter(x, threshold, scale):
    """Simple frequency domain filtering - scales low frequencies."""
    dtype = x.dtype
    x = x.type(torch.float32)

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)

    crow, ccol = H // 2, W // 2
    top = max(0, crow - threshold)
    bottom = min(H, crow + threshold)
    left = max(0, ccol - threshold)
    right = min(W, ccol + threshold)

    mask[..., top:bottom, left:right] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    x_filtered = x_filtered.type(dtype)

    return x_filtered


def fourier_solo(x, global_scale=1.0, freq_threshold=0, lf_scale=1.0, hf_scale=1.0,
                 amplitude_scale=1.0, phase_scale=1.0, blend_type=0):
    
    dtype = x.dtype
    x = x.type(torch.float32)
    
    # Apply global scale
    x = x * global_scale

    # Early return if no FFT processing needed
    if blend_type == -1:
        return x.type(dtype)

    # Perform FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    
    # Calculate center and bounds for low-frequency region
    crow, ccol = H // 2, W // 2
    top = max(0, crow - freq_threshold)
    bottom = min(H, crow + freq_threshold)
    left = max(0, ccol - freq_threshold)
    right = min(W, ccol + freq_threshold)

    
    
    # Extract amplitude and phase from the original frequency domain
    amplitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)

    # Apply different scaling strategies based on blend_type
    if blend_type == 0:
        # Scale all frequencies uniformly (amplitude and phase)
        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale
    elif blend_type == 1:
        # Scale only low frequencies
        amplitude_lf = amplitude[..., top:bottom, left:right] * amplitude_scale
        phase_lf = phase[..., top:bottom, left:right] * phase_scale
        amplitude[..., top:bottom, left:right] = amplitude_lf
        phase[..., top:bottom, left:right] = phase_lf
    elif blend_type == 2:
        # Scale high frequencies only, preserve low frequencies unchanged
        # Save low freq first
        lf_amplitude = amplitude[..., top:bottom, left:right].clone()
        lf_phase = phase[..., top:bottom, left:right].clone()
        
        # Scale everything
        amplitude = amplitude * amplitude_scale
        phase = phase * phase_scale
        
        # Restore low freq
        amplitude[..., top:bottom, left:right] = lf_amplitude
        phase[..., top:bottom, left:right] = lf_phase
    elif blend_type == 3:
        # Scale low freq amplitude, high freq phase
        amplitude_lf = amplitude[..., top:bottom, left:right] * amplitude_scale
        amplitude[..., top:bottom, left:right] = amplitude_lf
        
        lf_phase = phase[..., top:bottom, left:right].clone()
        phase = phase * phase_scale
        phase[..., top:bottom, left:right] = lf_phase
    elif blend_type == 4:
        # Scale high freq amplitude, low freq phase
        lf_amplitude = amplitude[..., top:bottom, left:right].clone()
        amplitude = amplitude * amplitude_scale
        amplitude[..., top:bottom, left:right] = lf_amplitude
        
        phase_lf = phase[..., top:bottom, left:right] * phase_scale
        phase[..., top:bottom, left:right] = phase_lf

    # NOW apply frequency scaling if specified (lf_scale/hf_scale)
    # This is separate from amplitude/phase scaling
    if lf_scale != 1.0 or hf_scale != 1.0:
        freq_mask = torch.ones((B, C, H, W), device=x.device, dtype=torch.float32)
        freq_mask[...] = hf_scale
        freq_mask[..., top:bottom, left:right] = lf_scale
        amplitude = amplitude * freq_mask

    # Reconstruct complex frequency representation
    reals = amplitude * torch.cos(phase)
    imags = amplitude * torch.sin(phase)
    x_freq_new = torch.complex(reals, imags)

    # Inverse FFT
    x_freq_new = fft.ifftshift(x_freq_new, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq_new, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)

    return x_filtered


def _set_attrs(block, **kwargs):
    for k, v in kwargs.items():
        setattr(block, k, v)


def register_tune_upblock2d(model, types=0,
                            k1=0.5, b1=1.2, t1=1, s1=0.9,
                            k1_1=0.5, b1_1=1.2, t1_1=1, s1_1=0.9,
                            k2=0.5, b2=1.4, t2=1, s2=0.2,
                            k2_1=0.5, b2_1=1.4, t2_1=1, s2_1=0.2,
                            g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
                            blend1=0, blend2=0, blend1_1=0, blend2_1=0,
                            a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
                            p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
                            skips=0, tunes=0):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            skipping = 0
            finetune = 0

            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] *= self.s1
                        finetune += 1
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        if self.k1 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                global_scale=self.g1, freq_threshold=self.t1,
                                lf_scale=self.b1, hf_scale=self.s1,
                                amplitude_scale=self.a1, phase_scale=self.p1,
                                blend_type=self.blend1,
                            )
                        if self.k1_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                global_scale=self.g1_1, freq_threshold=self.t1_1,
                                lf_scale=self.b1_1, hf_scale=self.s1_1,
                                amplitude_scale=self.a1_1, phase_scale=self.p1_1,
                                blend_type=self.blend1_1,
                            )
                        finetune += 1
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                    global_scale=self.g1, freq_threshold=self.t1,
                                    lf_scale=self.b1, hf_scale=self.s1,
                                    amplitude_scale=self.a1, phase_scale=self.p1,
                                    blend_type=self.blend1,
                                )
                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                    global_scale=self.g1_1, freq_threshold=self.t1_1,
                                    lf_scale=self.b1_1, hf_scale=self.s1_1,
                                    amplitude_scale=self.a1_1, phase_scale=self.p1_1,
                                    blend_type=self.blend1_1,
                                )
                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                    global_scale=self.g2, freq_threshold=self.t2,
                                    lf_scale=self.b2, hf_scale=self.s2,
                                    amplitude_scale=self.a2, phase_scale=self.p2,
                                    blend_type=self.blend2,
                                )
                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                    global_scale=self.g2_1, freq_threshold=self.t2_1,
                                    lf_scale=self.b2_1, hf_scale=self.s2_1,
                                    amplitude_scale=self.a2_1, phase_scale=self.p2_1,
                                    blend_type=self.blend2_1,
                                )
                        finetune += 1

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    if is_torch_version(">=", "1.11.0"):
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                        )
                    else:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(resnet), hidden_states, temb
                        )
                else:
                    hidden_states = resnet(hidden_states, temb)

                skipping += 1

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)
            _set_attrs(
                upsample_block,
                types=types,
                k1=k1, b1=b1, t1=t1, s1=s1,
                k2=k2, b2=b2, t2=t2, s2=s2,
                k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
                k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
                g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
                blend1=blend1, blend2=blend2, blend1_1=blend1_1, blend2_1=blend2_1,
                a1=a1, a2=a2, a1_1=a1_1, a2_1=a2_1,
                p1=p1, p2=p2, p1_1=p1_1, p2_1=p2_1,
                skips=skips, tunes=tunes,
            )


def register_tune_crossattn_upblock2d(model, types=0,
                                      k1=0.5, b1=1.2, t1=1, s1=0.9,
                                      k1_1=0.5, b1_1=1.2, t1_1=1, s1_1=0.9,
                                      k2=0.5, b2=1.4, t2=1, s2=0.2,
                                      k2_1=0.5, b2_1=1.4, t2_1=1, s2_1=0.2,
                                      g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
                                      blend1=0, blend2=0, blend1_1=0, blend2_1=0,
                                      a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
                                      p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
                                      skips=0, tunes=0):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None,
                    encoder_hidden_states=None, cross_attention_kwargs=None,
                    upsample_size=None, attention_mask=None, encoder_attention_mask=None):
            skipping = 0
            finetune = 0

            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if self.types == 1:
                    if hidden_states.shape[1] == 1280:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k1)] *= self.b1
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t1, scale=self.s1)
                    if hidden_states.shape[1] == 640:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states = fourier_filter(res_hidden_states, threshold=self.t2, scale=self.s2)
                elif self.types == 2:
                    if skipping >= self.skips and finetune < self.tunes:
                        hidden_states[:, :int(hidden_states.shape[1] * self.k2)] *= self.b2
                        res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] *= self.s2
                        finetune += 1
                elif self.types == 4:
                    if skipping >= self.skips and finetune < self.tunes:
                        if self.k2 > 0.0:
                            hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                global_scale=self.g2, freq_threshold=self.t2,
                                lf_scale=self.b2, hf_scale=self.s2,
                                amplitude_scale=self.a2, phase_scale=self.p2,
                                blend_type=self.blend2,
                            )
                        if self.k2_1 > 0.0:
                            res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                global_scale=self.g2_1, freq_threshold=self.t2_1,
                                lf_scale=self.b2_1, hf_scale=self.s2_1,
                                amplitude_scale=self.a2_1, phase_scale=self.p2_1,
                                blend_type=self.blend2_1,
                            )
                        finetune += 1
                elif self.types == 5:
                    if skipping >= self.skips and finetune < self.tunes:
                        if hidden_states.shape[1] == 1280:
                            if self.k1 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k1)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k1)],
                                    global_scale=self.g1, freq_threshold=self.t1,
                                    lf_scale=self.b1, hf_scale=self.s1,
                                    amplitude_scale=self.a1, phase_scale=self.p1,
                                    blend_type=self.blend1,
                                )
                            if self.k1_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k1_1)],
                                    global_scale=self.g1_1, freq_threshold=self.t1_1,
                                    lf_scale=self.b1_1, hf_scale=self.s1_1,
                                    amplitude_scale=self.a1_1, phase_scale=self.p1_1,
                                    blend_type=self.blend1_1,
                                )
                        if hidden_states.shape[1] == 640:
                            if self.k2 > 0.0:
                                hidden_states[:, :int(hidden_states.shape[1] * self.k2)] = fourier_solo(
                                    hidden_states[:, :int(hidden_states.shape[1] * self.k2)],
                                    global_scale=self.g2, freq_threshold=self.t2,
                                    lf_scale=self.b2, hf_scale=self.s2,
                                    amplitude_scale=self.a2, phase_scale=self.p2,
                                    blend_type=self.blend2,
                                )
                            if self.k2_1 > 0.0:
                                res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)] = fourier_solo(
                                    res_hidden_states[:, :int(res_hidden_states.shape[1] * self.k2_1)],
                                    global_scale=self.g2_1, freq_threshold=self.t2_1,
                                    lf_scale=self.b2_1, hf_scale=self.s2_1,
                                    amplitude_scale=self.a2_1, phase_scale=self.p2_1,
                                    blend_type=self.blend2_1,
                                )
                        finetune += 1

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            return module(*inputs)
                        return custom_forward

                    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, **ckpt_kwargs
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states, encoder_hidden_states, None, None,
                        cross_attention_kwargs, attention_mask, encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )[0]

                skipping += 1

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward
    crossattn_count = 0
    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            crossattn_count += 1
            if crossattn_count == 3:
                upsample_block.forward = up_forward(upsample_block)
                _set_attrs(
                upsample_block,
                types=types,
                k1=k1, b1=b1, t1=t1, s1=s1,
                k2=k2, b2=b2, t2=t2, s2=s2,
                k1_1=k1_1, b1_1=b1_1, t1_1=t1_1, s1_1=s1_1,
                k2_1=k2_1, b2_1=b2_1, t2_1=t2_1, s2_1=s2_1,
                g1=g1, g2=g2, g1_1=g1_1, g2_1=g2_1,
                blend1=blend1, blend2=blend2, blend1_1=blend1_1, blend2_1=blend2_1,
                a1=a1, a2=a2, a1_1=a1_1, a2_1=a2_1,
                p1=p1, p2=p2, p1_1=p1_1, p2_1=p2_1,
                skips=skips, tunes=tunes,
            )


def _load_pipeline(model_id, device, dtype, disable_safety_checker=False):
    kwargs = {"torch_dtype": dtype}

    if disable_safety_checker:
        kwargs.update({"safety_checker": None, "feature_extractor": None})

    pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

    if disable_safety_checker:
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

    pipe = pipe.to(device)
    return pipe


def _apply_dmfft(pipe, params):
    register_tune_upblock2d(pipe, **params)
    register_tune_crossattn_upblock2d(pipe, **params)


def main():
    parser = argparse.ArgumentParser(description="DMFFT - Fixed FFT implementation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=880)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="dmfft_fixed.png")
    parser.add_argument("--out-baseline", default="baseline_fixed.png")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--disable-safety-checker", action="store_true")
    parser.add_argument("--types", type=int, default=None, help="Override DMFFT type (1, 2, 4, 5)")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading model {args.model} on {device}...")
    pipe = _load_pipeline(
        args.model,
        device,
        dtype,
        disable_safety_checker=args.disable_safety_checker,
    )

    # Baseline (no DMFFT)
    baseline_params = dict(
        types=0,  # 0 = disabled
        k1=0.0, b1=1.0, t1=0, s1=1.0,
        k1_1=0.0, b1_1=1.0, t1_1=0, s1_1=1.0,
        k2=0.0, b2=1.0, t2=0, s2=1.0,
        k2_1=0.0, b2_1=1.0, t2_1=0, s2_1=1.0,
        g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
        blend1=0, blend2=0, blend1_1=0, blend2_1=0,
        a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
        p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
        skips=0, tunes=0,
    )

    # DMFFT params - using conservative settings from the paper
    # Type 1 is simpler and more stable than type 4
    dmfft_params = dict(
        types=1,  # Use type 1 (simpler frequency filtering)
        k1=0.5, b1=1.2, t1=1, s1=0.9,
        k1_1=0.5, b1_1=1.2, t1_1=1, s1_1=0.9,
        k2=0.5, b2=1.2, t2=1, s2=0.8,
        k2_1=0.5, b2_1=1.2, t2_1=1, s2_1=0.8,
        g1=1.0, g2=1.0, g1_1=1.0, g2_1=1.0,
        blend1=0, blend2=0, blend1_1=0, blend2_1=0,
        a1=1.0, a2=1.0, a1_1=1.0, a2_1=1.0,
        p1=1.0, p2=1.0, p1_1=1.0, p2_1=1.0,
        skips=0, tunes=10,
    )

    if args.types is not None:
        dmfft_params["types"] = int(args.types)
        print(f"Using custom DMFFT type: {args.types}")

    # Generate baseline
    if not args.no_baseline:
        print("Generating baseline image...")
        _apply_dmfft(pipe, baseline_params)
        torch.manual_seed(args.seed)
        base_img = pipe(args.prompt, num_inference_steps=args.steps).images[0]
        base_img.save(os.path.abspath(args.out_baseline))
        print(f"Baseline saved to {args.out_baseline}")

    # Generate DMFFT
    print("Generating DMFFT enhanced image...")
    _apply_dmfft(pipe, dmfft_params)
    torch.manual_seed(args.seed)
    dmfft_img = pipe(args.prompt, num_inference_steps=args.steps).images[0]
    dmfft_img.save(os.path.abspath(args.out))
    print(f"DMFFT image saved to {args.out}")


if __name__ == "__main__":
    main()