"""
Frozen V-JEPA 2.1 encoder wrapper.

Loads a pretrained V-JEPA 2.1 ViT encoder, discards the predictor,
and exposes a clean ``(clip) -> pooled_features`` (or patch tokens) interface.

Handles the ``src`` package name collision between vjepa2 and movad
by temporarily swapping cached ``src.*`` modules during import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_FILE = Path(__file__).resolve()
_VJEPA2_ROOT = _THIS_FILE.parent.parent / "vjepa2"

# ---------------------------------------------------------------------------
# Resolve the src-package collision between the two repos.
# ---------------------------------------------------------------------------
_SAVED_SRC_MODULES = {}
for _name in list(sys.modules.keys()):
    if _name == "src" or _name.startswith("src."):
        _SAVED_SRC_MODULES[_name] = sys.modules.pop(_name)

_MOVAD_SRC_DIR = _THIS_FILE.parent.parent / "src"
_STRIPPED_PATH = []
for _p in sys.path:
    _candidate = Path(_p) / "src"
    if _candidate.is_dir() and not (_candidate / "__init__.py").exists():
        _STRIPPED_PATH.append(_p)
for _p in _STRIPPED_PATH:
    sys.path.remove(_p)

sys.path.insert(0, str(_VJEPA2_ROOT))
try:
    from app.vjepa_2_1.models.vision_transformer import (  # noqa: E402
        VisionTransformer,
        vit_base,
        vit_large,
        vit_giant_xformers,
    )
finally:
    sys.path.remove(str(_VJEPA2_ROOT))
    for _p in reversed(_STRIPPED_PATH):
        sys.path.insert(0, _p)
    for _name, _mod in _SAVED_SRC_MODULES.items():
        sys.modules[_name] = _mod

# Helpers -------------------------------------------------------------------

_MODEL_BUILDERS = {
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_giant_xformers": vit_giant_xformers,
}


def load_pretrained_encoder(
    model_name: str = "vit_large",
    img_size: int = 256,
    num_frames: int = 16,
    tubelet_size: int = 2,
    patch_size: int = 16,
    checkpoint_path: str | None = None,
    checkpoint_key: str = "ema_encoder",
    device: str | torch.device = "cuda",
    **encoder_kwargs,
) -> VisionTransformer:
    """Build a V-JEPA 2.1 ViT and load pretrained weights."""
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model_name {model_name!r}. Choose from {list(_MODEL_BUILDERS)}")

    builder = _MODEL_BUILDERS[model_name]

    _defaults = dict(
        use_rope=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
    )
    _defaults.update(encoder_kwargs)

    encoder = builder(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        **_defaults,
    )

    if checkpoint_path is not None:
        print(f"Loading V-JEPA 2.1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint[checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # Try strict first — fail loudly if keys don't match exactly.
        # This ensures we never silently substitute random-init params.
        try:
            missing, unexpected = encoder.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                print(f"  strict load: missing={missing}, unexpected={unexpected}")
            print("  loaded with strict=True")
        except RuntimeError as e:
            print(f"  strict load failed: {e}")
            print("  falling back to shape-matched loading …")
            encoder_state = encoder.state_dict()
            for k, v in encoder_state.items():
                if k not in state_dict:
                    print(f'    key "{k}" not found in checkpoint — keeping random init')
                elif state_dict[k].shape != v.shape:
                    print(f'    key "{k}" shape mismatch: checkpoint {state_dict[k].shape} vs model {v.shape} — keeping model shape')
                    state_dict[k] = v
            msg = encoder.load_state_dict(state_dict, strict=False)
            print(f"  loaded with msg: {msg}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


class VJEPA2Encoder(nn.Module):
    """
    Frozen V-JEPA 2.1 encoder.

    Input
    -----
    x : ``[B, 3, T, H, W]``  video clip.

    Output (when ``return_patches=False``)
    --------------------------------------
    ``[B, embed_dim]``  spatially mean-pooled feature vector.

    Output (when ``return_patches=True``)
    -------------------------------------
    ``[B, N, embed_dim]``  full patch tokens (for SlotSSM cross-attention).
    """

    def __init__(self, encoder: VisionTransformer, pool: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.embed_dim: int = encoder.embed_dim
        self.pool = pool
        # Register encoder so .to(device)/.to(dtype) propagate correctly.
        self.add_module("encoder", encoder)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        z = self.encoder(x, training=self.training)  # [B, N, embed_dim]
        if return_patches:
            return z
        if self.pool == "mean":
            return z.mean(dim=1)
        if self.pool == "cls":
            return z[:, 0, :]
        raise ValueError(f"Unknown pool mode: {self.pool}")


def build_vjepa2_encoder(cfg) -> VJEPA2Encoder:
    """Build a VJEPA2Encoder from a MOVAD-style EasyDict config."""
    raw_encoder = load_pretrained_encoder(
        model_name=cfg.get("model_name", "vit_large"),
        img_size=cfg.get("img_size", 256),
        num_frames=cfg.get("num_frames", 16),
        tubelet_size=cfg.get("tubelet_size", 2),
        patch_size=cfg.get("patch_size", 16),
        checkpoint_path=cfg.get("checkpoint_path", None),
        checkpoint_key=cfg.get("checkpoint_key", "ema_encoder"),
        device=cfg.get("device", "cuda"),
    )
    return VJEPA2Encoder(raw_encoder, pool=cfg.get("pool", "mean"))
