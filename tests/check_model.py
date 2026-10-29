"""Released weights load strictly into aurora.model and match the reference pipeline.py outputs."""
import importlib.util, sys, torch
sys.path.insert(0, ".")
from aurora.model import build_models
from train import load_hf, param_groups, get_args

import types, inspect, functools
from types import SimpleNamespace
from torch import nn


def _stub_diffusers():
    """Minimal stand-in for diffusers, so the test runs without a working diffusers/xformers install."""
    def register_to_config(init):
        @functools.wraps(init)
        def wrapped(self, *a, **kw):
            bound = inspect.signature(init).bind(self, *a, **kw)
            bound.apply_defaults()
            init(self, *a, **kw)
            self.config = SimpleNamespace(**{k: v for k, v in bound.arguments.items() if k != "self"})
        return wrapped
    d = types.ModuleType("diffusers")
    d.ConfigMixin, d.ModelMixin, d.DiffusionPipeline = object, nn.Module, object
    mods = {"diffusers": d, "diffusers.configuration_utils": types.ModuleType("c"),
            "diffusers.pipelines": types.ModuleType("p"), "diffusers.pipelines.pipeline_utils": types.ModuleType("pu"),
            "diffusers.utils": types.ModuleType("u"), "diffusers.utils.torch_utils": types.ModuleType("tu")}
    mods["diffusers.configuration_utils"].register_to_config = register_to_config
    mods["diffusers.pipelines.pipeline_utils"].ImagePipelineOutput = object
    mods["diffusers.utils.torch_utils"].randn_tensor = None
    sys.modules.update(mods)


_stub_diffusers()

REF = "AuroraIMG2-50M-Preview"
spec = importlib.util.spec_from_file_location("ref", f"{REF}/pipeline.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

te, den = build_models()
n_te, n_den = sum(p.numel() for p in te.parameters()), sum(p.numel() for p in den.parameters())
print(f"text encoder {n_te/1e6:.2f}M (README 12.6M), denoiser {n_den/1e6:.2f}M (README 50.4M)")
load_hf(te, den, REF)  # strict
from safetensors.torch import load_file
rte, rden = ref.AuroraTextEncoder(), ref.AuroraDenoiser()
rte.load_state_dict(load_file(f"{REF}/text_encoder/diffusion_pytorch_model.safetensors"))
rden.load_state_dict(load_file(f"{REF}/transformer/diffusion_pytorch_model.safetensors"))
dev = "cuda"
for m in (te, den, rte, rden): m.to(dev).eval()
torch.manual_seed(0)
ids = torch.randint(6, 8192, (4, 64), device=dev); ids[:, 0] = 2; ids[:, 30] = 3; ids[:, 31:] = 0
z = torch.randn(4, 128, 16, 16, device=dev); t = torch.rand(4, device=dev)
with torch.no_grad():
    a, av = te(ids); b, bv = rte(ids)
    va = den(z, t, a, av); vb = rden(z, t, b, bv)
print("text max abs diff", (a - b).abs().max().item(), "velocity max abs diff", (va - vb).abs().max().item())
assert torch.equal(a, b) and torch.equal(va, vb)

sys.argv = ["x", "--data", "x", "--out", "x", "--no-compile"]
_, counts = param_groups(te, den, get_args())
print("optimizer groups:", {k: f"{v/1e6:.2f}M" for k, v in counts.items()})
print("OK")
