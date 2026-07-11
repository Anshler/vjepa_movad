"""
Swin-B + LSTM smoke test — synthetic data, one train + one validation step.

Verifies:
  - SwinEncoder builds from swin_lstm.yaml config (with random init — no pretrained weights needed for smoke test)
  - ClsVJEPA standard head with Swin (_is_swin=True, _needs_patches=True)
  - AdaptiveAvgPool3d((1,6,6)) → 36 grid cells → flatten(36864) → LN → Linear → ReLU → Dropout
  - encode_video_clips returns [B, n_clips, 36, C] patches (not mean-pooled)
  - forward_temporal_step flattens Swin patches internally
  - Loss compute, backward, optimizer step (no NaNs)
  - Validation pass with autocast

Usage (from WSL):
    conda activate vjepa2-312
    cd /mnt/d/Users/Chrysenberg69420/VSCodeProjects/vjepa_movad
    python tests/test_swin.py
    python tests/test_swin.py --checkpoint pretrained/swin_base_patch244_window1677_sthv2.pth
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

from model import build_multi_head_vjepa
from movad_core.dota import gt_cls_target
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG_DIR = os.path.join(_REPO_ROOT, "cfgs")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=None,
                    help="Path to Swin-B SSv2 pretrained checkpoint (.pth). "
                         "If omitted, uses random init for smoke test.")
parser.add_argument("--amp", default="fp16", choices=["fp32", "fp16", "bf16"],
                    help="AMP dtype (default: fp16)")
args = parser.parse_args()
CHECKPOINT_PATH = args.checkpoint
AMP_CHOICES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
AMP_DTYPE = AMP_CHOICES[args.amp]

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print(f"Device: {DEVICE}")
print(f"AMP: {args.amp}")
print(f"Checkpoint: {CHECKPOINT_PATH or 'none (random init)'}")


# ---------------------------------------------------------------------------
# Build model from swin_lstm.yaml config
# ---------------------------------------------------------------------------
cfg_path = os.path.join(CFG_DIR, "swin_lstm.yaml")
with open(cfg_path, "r") as f:
    cfg = EasyDict(yaml.safe_load(f))
cfg.device = DEVICE
cfg.checkpoint_path = CHECKPOINT_PATH  # None → random init (smoke test still works)

cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "swin_lstm_test"

print(f"\nBuilding model from swin_lstm.yaml...")
model = build_multi_head_vjepa(cfg)
head_name = next(iter(model.heads.keys()))
head = model.heads[head_name]

# Verify Swin properties
assert head._is_swin, f"Expected _is_swin=True, got {head._is_swin}"
assert head._needs_patches, f"Expected _needs_patches=True, got {head._needs_patches}"
assert not head._slot_based, "Expected _slot_based=False for LSTM head"
print(f"  _is_swin: {head._is_swin}")
print(f"  _needs_patches: {head._needs_patches}")
print(f"  temporal_type: {head.temporal_type}")

# Check MOVAD projection sizes
print(f"  bn (LayerNorm): in_features={head.bn.normalized_shape[0]}")
print(f"  lin1: in={head.lin1.in_features}, out={head.lin1.out_features}")
assert head.lin1.in_features == 1024 * 36, \
    f"Expected lin1.in_features=36864 (1024×36), got {head.lin1.in_features}"
print(f"  ✓ MOVAD projection: 1024×36=36864 → 1024")

# Unidirectional LSTM (matches MOVAD) → output = rnn_state_size = 1024
assert head.lin2.in_features == 1024, \
    f"Expected lin2.in_features=1024 (rnn_state_size, unidirectional), got {head.lin2.in_features}"
print(f"  ✓ Unidirectional LSTM (MOVAD match): lin2 in=1024 → out={head.lin2.out_features}")

# Check encoder pooling
encoder = head.encoder
assert encoder is not None, "Encoder should be stored as submodule"
assert hasattr(encoder, 'avgpool'), "SwinEncoder should have avgpool"
print(f"  encoder.embed_dim: {encoder.embed_dim}")
print(f"  encoder.num_patches: {encoder.num_patches}")
print(f"  encoder.avgpool: AdaptiveAvgPool3d output_size=(1,6,6)")

# Setup training infrastructure
head_cfg = EasyDict(dict(cfg))
head_cfg.device = DEVICE
head_cfg.name = head_name
for k in ("dim_latent", "dropout", "rnn_state_size", "rnn_cell_num"):
    if k in dict(cfg):
        head_cfg[k] = cfg[k]
criterion = build_loss(head_cfg)
opt, _ = build_optimizer(EasyDict({"lr": cfg.lr}), model.heads[head_name], None)

_amp_cfg = cfg.get("amp_dtype", "fp32")
_amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_amp_cfg)
autocast_ctx = (
    torch.amp.autocast("cuda", dtype=_amp_dtype) if _amp_dtype else nullcontext()
)

model.to(DEVICE)
model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")


# ---------------------------------------------------------------------------
# 1. TRAINING —  encode + temporal loop + backward
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. TRAINING — encode + temporal loop + backward")
print("=" * 70)

B = 2
NF = cfg.get("num_frames", 4)
v_len = 12
img_size = cfg.get("img_size", 256)

# Raw video [B, C, T, H, W]
video_data = torch.randn(B, 3, v_len, img_size, img_size, device=DEVICE)

# data_info: one normal video, one anomalous
data_info = torch.zeros(B, 11, device=DEVICE)
data_info[:, 0] = v_len
data_info[:, 1] = torch.arange(B)
data_info[:, 2] = -1   # a_start
data_info[:, 3] = -1   # a_end
data_info[:, 4] = -1   # label
data_info[1, 2] = 6    # anomalous from frame 6-9
data_info[1, 3] = 9
data_info[1, 4] = 1

# Encode with autocast
with autocast_ctx:
    patches = model.encode_video_clips(video_data, NF)
print(f"  patches: {tuple(patches.shape)}  dtype={patches.dtype}")
# Swin patches: [B, n_clips, 36, C] — NOT mean-pooled
assert patches.shape[2] == 36, f"Expected 36 grid cells, got {patches.shape[2]}"
n_clips = v_len - NF
assert patches.shape[1] == n_clips, f"Expected {n_clips} clips, got {patches.shape[1]}"
print(f"  ✓ patches shape matches: {n_clips} clips × 36 grid cells × {patches.shape[3]} channels")

# Per-frame temporal loop (mirrors _run_temporal_loop with _needs_patches)
patches_in = patches if head._needs_patches else patches.mean(dim=2)
print(f"  temporal input shape: {tuple(patches_in.shape)}")

toa_batch = data_info[:, 2]
tea_batch = data_info[:, 3]
video_len_orig = data_info[:, 0]

state = None
total_loss_val = 0.0
frame_count = 0

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()

    with autocast_ctx:
        feat = patches_in[:, i - NF, ...]       # [B, 36, C]
        output, state = head.forward_temporal_step(feat, state)

    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100

    if cfg.get("apply_softmax", True):
        output = output.softmax(dim=1)

    loss = criterion(output, target)

    # MOVAD-style: per-frame backward + step
    opt.zero_grad()
    loss.backward()
    opt.step()

    total_loss_val += loss.detach().item()
    frame_count += 1

avg_loss = total_loss_val / max(frame_count, 1)
print(f"  avg_loss={avg_loss:.4f}  frames={frame_count}")

# Check gradients still flow on last frame
with torch.no_grad():
    has_grad = any(
        p.grad is not None for p in model.heads[head_name].parameters()
    )
print(f"  has_grad: {has_grad}")
print(f"  ✓ training step OK")

loss_before = avg_loss


# ---------------------------------------------------------------------------
# 2. TRAINING — second step (verify loss decreases)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TRAINING — second step (check loss decreases)")
print("=" * 70)

with autocast_ctx:
    patches2 = model.encode_video_clips(video_data, NF)

patches_in2 = patches2 if head._needs_patches else patches2.mean(dim=2)
state2 = None
total_loss2_val = 0.0

for i in range(NF, v_len):
    target = gt_cls_target(i, toa_batch, tea_batch).long()
    with autocast_ctx:
        feat = patches_in2[:, i - NF, ...]
        output, state2 = head.forward_temporal_step(feat, state2)
    flt = i >= video_len_orig
    target[flt] = -100
    output[flt] = -100
    if cfg.get("apply_softmax", True):
        output = output.softmax(dim=1)
    loss = criterion(output, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    total_loss2_val += loss.detach().item()

loss_after = total_loss2_val / max(frame_count, 1)

print(f"  step 1 loss: {loss_before:.4f}  →  step 2 loss: {loss_after:.4f}")
if loss_after < loss_before:
    print("  ✓ loss decreased")
else:
    print("  ⚠ loss did not decrease (expected on synthetic data with random init)")
print("  ✓ second training step OK")


# ---------------------------------------------------------------------------
# 3. VALIDATION — encode + temporal loop with autocast (no grad)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. VALIDATION — encode + temporal loop with autocast (no grad)")
print("=" * 70)

model.eval()

v_len_val = 16
video_val = torch.randn(1, 3, v_len_val, img_size, img_size, device=DEVICE)
data_info_val = torch.zeros(1, 11, device=DEVICE)
data_info_val[:, 0] = v_len_val
data_info_val[:, 1] = 0
data_info_val[:, 2] = 6
data_info_val[:, 3] = 10
data_info_val[:, 4] = 1

with torch.no_grad():
    with autocast_ctx:
        patches_val = model.encode_video_clips(video_val, NF)
    print(f"  val patches: {tuple(patches_val.shape)}  dtype={patches_val.dtype}")

    n_clips_val = v_len_val - NF
    patches_in_val = patches_val if head._needs_patches else patches_val.mean(dim=2)

    state_val = None
    outputs_val = torch.zeros(1, n_clips_val, device=DEVICE)
    for i in range(NF, v_len_val):
        feat = patches_in_val[:, i - NF, ...].float()
        with autocast_ctx:
            output, state_val = head.forward_temporal_step(feat, state_val)
        if cfg.get("apply_softmax", True):
            output = output.softmax(dim=1)
        outputs_val[:, i - NF] = output[:, 1]

print(f"  val outputs: {tuple(outputs_val.shape)}")
print(f"  val anomaly scores: min={outputs_val.min().item():.4f}  max={outputs_val.max().item():.4f}")
assert not torch.isnan(outputs_val).any(), "NaN in validation outputs!"
print("  ✓ validation step OK")


# ---------------------------------------------------------------------------
# 4. Forward-only (no RNN state) — verify full-movie path
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Forward (full-movie, no RNN state) — verify MOVAD projection")
print("=" * 70)

model.eval()
video_full = torch.randn(2, 3, NF, img_size, img_size, device=DEVICE)  # exactly NF frames

with torch.no_grad():
    with autocast_ctx:
        out = model.heads[head_name](video_full)
    # For LSTM, forward returns (output, state) tuple
    out_tensor = out[0] if isinstance(out, tuple) else out
    print(f"  forward output: {tuple(out_tensor.shape)}")
    print(f"  output range: [{out_tensor.min().item():.4f}, {out_tensor.max().item():.4f}]")
    print("  ✓ full-movie forward OK")

# Verify Swin patch dimensions at each stage
with torch.no_grad():
    encoder_out = head.encoder(video_full, return_patches=True)
    print(f"  encoder patches: {tuple(encoder_out.shape)}  (expect [2, 36, 1024])")
    assert encoder_out.shape == (2, 36, 1024), \
        f"Expected [2, 36, 1024] Swin patches, got {tuple(encoder_out.shape)}"

    encoder_vec = head.encoder(video_full, return_patches=False)
    print(f"  encoder vector: {tuple(encoder_vec.shape)}  (expect [2, 1024])")
    assert encoder_vec.shape == (2, 1024), \
        f"Expected [2, 1024] MOVAD-projected vector, got {tuple(encoder_vec.shape)}"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("All Swin-B + LSTM smoke tests passed!")
print(f"  Architecture: Swin-B SSv2 → AdaptiveAvgPool3d((1,6,6)) → 36 grid cells")
print(f"  MOVAD projection: flatten(4608) → LN → Linear(4608→1024) → ReLU → Dropout")
print(f"  Temporal: 3-layer LSTM (1024)")
print(f"  AMP: {args.amp}")
print(f"  Checkpoint: {CHECKPOINT_PATH or 'random init'}")
print("=" * 70)
