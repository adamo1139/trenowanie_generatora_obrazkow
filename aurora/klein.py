"""FLUX.2 [klein] base 4B as a frozen teacher, in this repo's conventions.

klein works in the same latent space as our denoiser: FLUX.2 VAE latents, 2x2-packed to 128 channels and normalised
with the VAE's batch-norm statistics (B, 128, h, w), rectified flow z_t = (1 - t) x + t eps with t = sigma, velocity
eps - x (diffusers' FlowMatchEulerDiscreteScheduler: x_next = x + (sigma_next - sigma) v). So its velocity can be used
as a target for ours directly; only the text side differs: Qwen3-4B with the chat template, the hidden states of
layers 9 / 18 / 27 stacked to width 7680, padded to 512 tokens (no mask; that is how klein was trained), and 4D (t, h,
w, l) positions for text and image tokens. Mirrors diffusers' Flux2KleinPipeline (0.42).
"""
import math

import torch

REPO = "black-forest-labs/FLUX.2-klein-base-4B"


def klein_sigmas(steps: int, tokens: int, device=None) -> torch.Tensor:
    """The pipeline's sampling schedule: linspace(1, 1/steps) shifted by its resolution / step-count dependent mu
    (compute_empirical_mu, exponential time shift), then a final 0. -> (steps + 1,)"""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if tokens > 4300:
        mu = a2 * tokens + b2
    else:
        m200, m10 = a2 * tokens + b2, a1 * tokens + b1
        a = (m200 - m10) / 190.0
        mu = a * steps + (m200 - 200.0 * a)
    s = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    return torch.cat([s, s.new_zeros(1)]).float().to(device)


class KleinTeacher:
    """Frozen klein transformer (+ its Qwen3 text encoder for encode()); plain object, never trained."""

    def __init__(self, device, repo: str = REPO, dtype=torch.bfloat16, text_encoder: bool = True):
        from diffusers import Flux2Transformer2DModel
        self.device, self.dtype = device, dtype
        self.transformer = Flux2Transformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=dtype)
        self.transformer.to(device).eval().requires_grad_(False)
        self.te = self.tok = None
        if text_encoder:
            from transformers import AutoTokenizer, Qwen3ForCausalLM
            self.tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
            self.te = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype)
            self.te.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, prompts, max_len: int = 512, layers=(9, 18, 27)) -> torch.Tensor:
        """Prompts -> (B, max_len, 7680) text states, as Flux2KleinPipeline._get_qwen3_prompt_embeds."""
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        out = self.te(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device),
                      output_hidden_states=True, use_cache=False)
        h = torch.stack([out.hidden_states[k] for k in layers], dim=1)  # (B, 3, L, 2560)
        b, c, l, d = h.shape
        return h.permute(0, 2, 1, 3).reshape(b, l, c * d).to(self.dtype)

    @staticmethod
    def _ids(b, h, w, l_txt, device):
        img = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1))
        txt = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(l_txt))
        return img.to(device).expand(b, -1, -1), txt.to(device).expand(b, -1, -1)

    @torch.no_grad()
    def velocity(self, x, t, text, null_text=None, cfg: float = 4.0) -> torch.Tensor:
        """x: (B, 128, h, w) normalised latents at time t (B,) in [0, 1]; text: (B, L, 7680); null_text: (1 or B, L,
        7680) states of the empty prompt (the pipeline's negative prompt) for CFG. -> (B, 128, h, w) fp32 velocity
        eps - x (guided: v_null + cfg (v_text - v_null))."""
        b, c, h, w = x.shape
        z = x.to(self.dtype).reshape(b, c, h * w).transpose(1, 2)
        guided = null_text is not None and cfg != 1.0
        if guided:
            z = torch.cat([z, z])
            text = torch.cat([text, null_text.expand(b, -1, -1).to(text.dtype)])
            t = torch.cat([t, t])
        img_ids, txt_ids = self._ids(z.shape[0], h, w, text.shape[1], x.device)
        v = self.transformer(hidden_states=z, timestep=t.to(self.dtype), guidance=None, encoder_hidden_states=text,
                             txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0].float()
        if guided:
            v_text, v_null = v.chunk(2)
            v = v_null + cfg * (v_text - v_null)
        return v.transpose(1, 2).reshape(b, c, h, w)

    @torch.no_grad()
    def sample(self, text, null_text, noise, steps: int = 50, cfg: float = 4.0) -> torch.Tensor:
        """Euler with the pipeline's schedule, from `noise` (B, 128, h, w). -> normalised latents."""
        sig = klein_sigmas(steps, noise.shape[2] * noise.shape[3], noise.device)
        z = noise.float()
        for k in range(steps):
            v = self.velocity(z, sig[k].expand(z.shape[0]), text, null_text, cfg)
            z = z + (sig[k + 1] - sig[k]) * v
        return z
