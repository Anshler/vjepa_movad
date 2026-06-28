"""
Quick diagnostic: run SlotSSM + inverted cross-attn for a few steps
with random video input and report per-slot mass to detect dead slots.

Usage:  python tests/diag_slots.py
"""
from __future__ import annotations

import sys, os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml
from easydict import EasyDict

from model import build_cls_vjepa

# ---------------------------------------------------------------------------
# Build SlotSSM with inverted attention via the config path
# ---------------------------------------------------------------------------
cfg = EasyDict(yaml.safe_load(open(os.path.join(_REPO_ROOT, "cfgs", "vjepa_slotssm_inv.yaml"))))
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.checkpoint_path = None          # random init encoder — ok for diagnostics
cfg.compile = False                 # skip torch.compile for quick test
print(f"Device: {cfg.device}")
print(f"Temporal: {cfg.temporal_model}  inverted_attn={cfg.use_inverted_attention}")

model = build_cls_vjepa(cfg)
model.eval()

# ---------------------------------------------------------------------------
# Feed 4 random 16-frame clips, print slot cross-attn diagnostics
# ---------------------------------------------------------------------------
print("\n{:>4s}  {:>10s}  {:>10s}  {:>10s}".format("Step", "mass_min", "mass_mean", "usage_frac"))
print("-" * 52)

for step in range(4):
    x = torch.randn(8, 3, 16, 256, 256, device=cfg.device)  # [B, C, T, H, W]
    with torch.no_grad():
        _, _ = model(x, state=None)

    temporal = model.temporal
    mass_min = float(temporal._slot_mass_min)
    mass_mean = float(temporal._slot_mass_mean)
    usage = float(temporal._slot_usage_frac)

    print(f"{step:>4d}  {mass_min:>10.6f}  {mass_mean:>10.6f}  {usage:>10.4f}")

print("\nKey: mass_min = lowest per-slot mass (fraction of fair share, 1.0=even)")
print("       mass_mean = avg mass across 32 slots (should be ~1.0)")
print("       usage_frac = fraction of slots >= 15% of fair share (dead if << 1.0)")
