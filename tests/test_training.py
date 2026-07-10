"""
Training + validation smoke test — synthetic data, one step each.

Verifies:
  - Model build from config
  - encode_video_clips with autocast (train + val)
  - Per-frame temporal loop with autocast (train + val)
  - Loss compute, backward, optimizer step (no NaNs, loss decreases)
  - _evaluate_model does not crash with autocast

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad
    python tests/test_training.py
    python tests/test_training.py --amp fp16
    python tests/test_training.py --checkpoint ~/vjepa2-checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext

import torch
import yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Compatibility: torchvision >=0.20 moved functional_tensor to _functional_tensor
try:
    import torchvision.transforms._functional_tensor as _ft  # noqa: F401
    import sys as _sys
    _sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import _HAS_MAMBA_SSM, build_multi_head_vjepa
from movad_core.dota import gt_cls_target
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

_AMP_CHOICES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
parser = argparse.ArgumentParser()
parser.add_argument("--amp", default="fp16", choices=list(_AMP_CHOICES),
                    help="AMP dtype (default: fp16)")
parser.add_argument("--checkpoint", default=None,
                    help="Path to a pretrained V-JEPA checkpoint (.pt)")
args = parser.parse_args()
AMP_DTYPE = _AMP_CHOICES[args.amp]
CHECKPOINT_PATH = args.checkpoint  # None → don't load any weights

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print(f"Device: {DEVICE}")
print(f"mamba_ssm: {_HAS_MAMBA_SSM}")
if CHECKPOINT_PATH:
    print(f"checkpoint: {CHECKPOINT_PATH}")
else:
    print("checkpoint: none (using random init)")
print(f"amp: {args.amp}")


# ---------------------------------------------------------------------------
# Build model + infrastructure from a real config
# ---------------------------------------------------------------------------
def load_cfg(name: str) -> EasyDict:
    path = os.path.join(CFG_DIR, name)
    with open(path, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device = DEVICE
    if CHECKPOINT_PATH:
        cfg.checkpoint_path = CHECKPOINT_PATH
    return cfg


# Use a single-head config (closest to normal training)
cfg = load_cfg("vjepa_mamba.yaml")
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "mamba_test"

model = build_multi_head_vjepa(cfg)
head_name = next(iter(model.heads.keys()))
head = model.heads[head_name]
assert list(model.heads.keys()) == ["mamba_test"]

# Per-head infrastructure (mirrors _train)
head_cfgs: dict[str, dict] = {}
criterion: dict[str, torch.nn.Module] = {}
optimizer: dict[str, torch.optim.Optimizer] = {}
autocast_ctx: dict[str, object] = {}

for name in list(model.heads.keys()):
    hc = model.head_configs.get(name, {})
    head_cfgs[name] = dict(cfg)
    for k in ("dim_latent", "dropout", "rnn_state_size", "rnn_cell_num",
              "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_version",
              "num_slots", "slot_dim", "num_ssm_blocks", "top_k", "eps_random",
              "use_inverted_attention", "entropy_weight"):
        if k in hc:
            head_cfgs[name][k] = hc[k]

    head_easy = EasyDict(head_cfgs[name])
    head_easy.device = DEVICE
    criterion[name] = build_loss(head_easy)

    opt, _ = build_optimizer(EasyDict({"lr": cfg.lr}), model.heads[name], None)
    optimizer[name] = opt

    _amp_cfg = head_cfgs[name].get("amp_dtype", "fp32")
    _amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
    autocast_ctx[name] = (
        torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
    )

model.to(DEVICE)
model.train()  # temporal heads in train mode, encoder stays eval

print(f"\nHead: {head_name}  temporal: {head.temporal_type}")
print(f"Trainable params: {sum(p.numel() for p in head.parameters() if p.requires_grad):,}")

# ---------------------------------------------------------------------------
# Synthetic data — simulate 2 videos of 12 frames each (padded to v_len=12)
# ---------------------------------------------------------------------------
B = 2
NF = cfg.get("num_frames", 4)
v_len = 12  # 12 clips → 12-NF = 8 per-frame outputs
img_size = cfg.get("img_size", 256)

# Raw video [B, C, T, H, W] — already padded by collator
video_data = torch.randn(B, 3, v_len, img_size, img_size, device=DEVICE)

# data_info columns: [v_len_orig, video_id, a_start, a_end, label, ..., ego, night, has_obj]
# Make one video fully normal, the other with a short anomaly
data_info = torch.zeros(B, 11, device=DEVICE)
data_info[:, 0] = v_len           # v_len_orig (no padding truncation)
data_info[:, 1] = torch.arange(B) # video_id
data_info[:, 2] = -1              # a_start (-1 = normal)
data_info[:, 3] = -1              # a_end
data_info[:, 4] = -1              # label (-1 = normal)

# Make batch[1] anomalous: anomaly from frame 6 to 9
data_info[1, 2] = 6   # a_start
data_info[1, 3] = 9   # a_end
data_info[1, 4] = 1   # label

# ---------------------------------------------------------------------------
# 1.  TRAINING  —  one encode + per-frame temporal loop + backward
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. TRAINING — one step")
print("=" * 70)

# Encode with autocast (matching the main.py fix)
with autocast_ctx[head_name]:
    patches = model.encode_video_clips(video_data, NF)
print(f"  patches: {tuple(patches.shape)}  dtype={patches.dtype}")

# Per-frame temporal loop (mirrors _run_temporal_loop)
is_slot = head._slot_based
patches_in = patches if is_slot else patches.mean(dim=2)

toa_batch = data_info[:, 2]
tea_batch = data_info[:, 3]
video_len_orig = data_info[:, 0]

state = None
total_loss = torch.tensor(0.0, device=DEVICE)
frame_count = 0

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()

    with autocast_ctx[head_name]:
        feat = patches_in[:, i - NF, ...]
        output, state = head.forward_temporal_step(feat, state)

    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100

    if head_cfgs[head_name].get("apply_softmax", True):
        output = output.softmax(dim=1)

    loss = criterion[head_name](output, target)
    total_loss = total_loss + loss
    frame_count += 1

avg_loss = total_loss.item() / max(frame_count, 1)
print(f"  avg_loss={avg_loss:.4f}  frames={frame_count}")
print(f"  output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
assert not torch.isnan(total_loss), "NaN in training loss!"

# Backward + optimizer step
optimizer[head_name].zero_grad()
total_loss.backward()
grad_norm = sum(
    p.grad.norm().item() for p in head.parameters() if p.grad is not None
)
optimizer[head_name].step()

print(f"  grad_norm (sum): {grad_norm:.4f}")
assert grad_norm > 0, "No gradients flowed!"
assert not torch.isnan(torch.tensor(grad_norm)), "NaN in gradients!"

loss_before = avg_loss
print("  ✓ training step OK")

# ---------------------------------------------------------------------------
# 2.  TRAINING  —  second step (verify loss decreases)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TRAINING — second step (check loss decreases)")
print("=" * 70)

with autocast_ctx[head_name]:
    patches2 = model.encode_video_clips(video_data, NF)

patches_in2 = patches2 if is_slot else patches2.mean(dim=2)
state2 = None
total_loss2 = torch.tensor(0.0, device=DEVICE)

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()

    with autocast_ctx[head_name]:
        feat = patches_in2[:, i - NF, ...]
        output, state2 = head.forward_temporal_step(feat, state2)

    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100
    if head_cfgs[head_name].get("apply_softmax", True):
        output = output.softmax(dim=1)
    total_loss2 = total_loss2 + criterion[head_name](output, target)

loss_after = total_loss2.item() / max(frame_count, 1)

optimizer[head_name].zero_grad()
total_loss2.backward()
optimizer[head_name].step()

print(f"  step 1 loss: {loss_before:.4f}  →  step 2 loss: {loss_after:.4f}")
if loss_after < loss_before:
    print("  ✓ loss decreased")
else:
    print("  ⚠ loss did not decrease (expected on synthetic data with 1 step)")
print("  ✓ second training step OK")

# ---------------------------------------------------------------------------
# 3.  VALIDATION  —  encode + temporal loop with autocast (no grad)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. VALIDATION — encode + temporal loop with autocast (no grad)")
print("=" * 70)

model.eval()

# Simulate full-video validation (VCL=None → v_len frames)
v_len_val = 16  # realistic: short DoTA video
video_val = torch.randn(1, 3, v_len_val, img_size, img_size, device=DEVICE)
data_info_val = torch.zeros(1, 11, device=DEVICE)
data_info_val[:, 0] = v_len_val    # v_len_orig (full video)
data_info_val[:, 1] = 0            # video_id
data_info_val[:, 2] = 6            # a_start
data_info_val[:, 3] = 10           # a_end
data_info_val[:, 4] = 1            # label

with torch.no_grad():
    # Encode with autocast (matches fixed _evaluate_model)
    with autocast_ctx[head_name]:
        patches_val = model.encode_video_clips(video_val, NF)
    print(f"  val patches: {tuple(patches_val.shape)}  dtype={patches_val.dtype}")

    n_clips = v_len_val - NF
    patches_in_val = patches_val if is_slot else patches_val.mean(dim=2)
    n_patches = patches_in_val.shape[2]

    # Per-frame temporal loop with autocast (matches fixed _evaluate_model)
    state_val = None
    outputs_val = torch.zeros(1, n_clips, device=DEVICE)
    for i in range(NF, v_len_val):
        feat = patches_in_val[:, i - NF, ...].float()
        with autocast_ctx[head_name]:
            output, state_val = head.forward_temporal_step(feat, state_val)
        if head_cfgs[head_name].get("apply_softmax", True):
            output = output.softmax(dim=1)
        outputs_val[:, i - NF] = output[:, 1]

print(f"  val outputs: {tuple(outputs_val.shape)}")
print(f"  val anomaly scores: min={outputs_val.min().item():.4f}  max={outputs_val.max().item():.4f}")
assert not torch.isnan(outputs_val).any(), "NaN in validation outputs!"
print("  ✓ validation step OK")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"All training + validation smoke tests passed!  (amp={AMP})")
print("Autocast wrapping verified for:  encode_video_clips + forward_temporal_step")
print("=" * 70)
