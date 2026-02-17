import argparse
from typing import Any, Dict, Optional, Tuple

import torch
import torch.fft as fft
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_torch_version


def isinstance_str(x: object, cls_name: str) -> bool:
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def fourier_filter(x: torch.Tensor, threshold: int, scale: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.type(torch.float32)

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    top = 0 if crow - threshold < 0 else crow - threshold
    left = 0 if ccol - threshold < 0 else ccol - threshold

    mask[..., top:crow + threshold, left:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    x_filtered = x_filtered.type(dtype)

    return x_filtered


def _freeu_scale(h: torch.Tensor, b: float) -> torch.Tensor:
    hidden_mean = h.mean(1, keepdim=True)
    hidden_max = hidden_mean.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    hidden_min = hidden_mean.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    hidden_mean = (hidden_mean - hidden_min) / (hidden_max - hidden_min)
    h[:, : h.shape[1] // 2] = h[:, : h.shape[1] // 2] * ((b - 1) * hidden_mean + 1)
    return h


def register_freeu_upblock2d(model, b1=1.4, b2=1.6, s1=0.9, s2=0.2, ch1=1280, ch2=640, t1=1, t2=1):
    def up_forward(self):
        def forward(hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
            for resnet in self.resnets:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if hidden_states.shape[1] == ch1:
                    hidden_states = _freeu_scale(hidden_states, b1)
                    res_hidden_states = fourier_filter(res_hidden_states, threshold=t1, scale=s1)

                if hidden_states.shape[1] == ch2:
                    hidden_states = _freeu_scale(hidden_states, b2)
                    res_hidden_states = fourier_filter(res_hidden_states, threshold=t2, scale=s2)

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

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "UpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def register_freeu_crossattn_upblock2d(model, b1=1.4, b2=1.6, s1=0.9, s2=0.2, ch1=1280, ch2=640, t1=1, t2=1):
    def up_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
        ):
            for resnet, attn in zip(self.resnets, self.attentions):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                if hidden_states.shape[1] == ch1:
                    hidden_states = _freeu_scale(hidden_states, b1)
                    res_hidden_states = fourier_filter(res_hidden_states, threshold=t1, scale=s1)

                if hidden_states.shape[1] == ch2:
                    hidden_states = _freeu_scale(hidden_states, b2)
                    res_hidden_states = fourier_filter(res_hidden_states, threshold=t2, scale=s2)

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            return module(*inputs)
                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False),
                        hidden_states,
                        encoder_hidden_states,
                        None,
                        None,
                        cross_attention_kwargs,
                        attention_mask,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )[0]

            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    hidden_states = upsampler(hidden_states, upsample_size)

            return hidden_states

        return forward

    for upsample_block in model.unet.up_blocks:
        if isinstance_str(upsample_block, "CrossAttnUpBlock2D"):
            upsample_block.forward = up_forward(upsample_block)


def load_pipeline(model_id: str, device: str, dtype: torch.dtype) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


def make_generator(seed: int, device: str) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def apply_freeu(pipe: StableDiffusionPipeline, b1: float, b2: float, s1: float, s2: float, ch1: int, ch2: int, t1: int, t2: int) -> None:
    register_freeu_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2, ch1=ch1, ch2=ch2, t1=t1, t2=t2)
    register_freeu_crossattn_upblock2d(pipe, b1=b1, b2=b2, s1=s1, s2=s2, ch1=ch1, ch2=ch2, t1=t1, t2=t2)


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeU exact paper implementation")
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=880)
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|fp16|fp32")
    parser.add_argument("--out", type=str, default="freeu.png")
    parser.add_argument("--out-baseline", type=str, default="baseline.png")
    parser.add_argument("--no-baseline", action="store_true")

    # FreeU defaults for SD2.1 from official README
    parser.add_argument("--b1", type=float, default=1.4)
    parser.add_argument("--b2", type=float, default=1.6)
    parser.add_argument("--s1", type=float, default=0.9)
    parser.add_argument("--s2", type=float, default=0.2)

    parser.add_argument("--ch1", type=int, default=1280)
    parser.add_argument("--ch2", type=int, default=640)
    parser.add_argument("--t1", type=int, default=1)
    parser.add_argument("--t2", type=int, default=1)

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dtype == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = load_pipeline(args.model, device, dtype)

    if not args.no_baseline:
        img_base = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            generator=make_generator(args.seed, device),
        ).images[0]
        img_base.save(args.out_baseline)

    apply_freeu(pipe, args.b1, args.b2, args.s1, args.s2, args.ch1, args.ch2, args.t1, args.t2)
    img_freeu = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        generator=make_generator(args.seed, device),
    ).images[0]
    img_freeu.save(args.out)


if __name__ == "__main__":
    main()
