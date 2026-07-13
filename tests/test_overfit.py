"""
Fast single-config overfit test.
Usage:
    python tests/test_overfit.py --train_encoder --lr 0.005 --epochs 100
    python tests/test_overfit.py                    # frozen, lr=0.01
    python tests/test_overfit.py --train_encoder    # trainable, lr=0.01
    python tests/test_overfit.py --softmax          # with old double-softmax behavior
"""
from __future__ import annotations

import argparse, os, sys, numpy as np, torch, torch.nn.functional as F, yaml
from easydict import EasyDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _ft)
except ImportError:
    pass

from model import build_multi_head_vjepa
from movad_core.dota import gt_cls_target
from movad_core.losses import build_loss
from movad_core.optim import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="cfgs/vjepa_mamba.yaml",
                    help="Path to config YAML (e.g. cfgs/swin_lstm.yaml or cfgs/vjepa_v1.yaml)")
parser.add_argument("--checkpoint", default=None,
                    help="Override checkpoint path (uses config value if not set)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--train_encoder", action="store_true", default=False)
parser.add_argument("--softmax", action="store_true", default=False,
                    help="Enable double-softmax (for comparing with old behavior)")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
apply_softmax = args.softmax
print(f"Device: {DEVICE}  |  train_encoder={args.train_encoder}  |  lr={args.lr}  |  epochs={args.epochs}  |  softmax={apply_softmax}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# --- Build model ---
cfg_path = os.path.join(_REPO_ROOT, args.config)
with open(cfg_path, "r") as f:
    base_cfg = EasyDict(yaml.safe_load(f))
NF = base_cfg.get("num_frames", 4)
VCL = base_cfg.get("VCL", 8)
img_size = base_cfg.get("img_size", 256)

cfg = EasyDict(yaml.safe_load(open(cfg_path)))
cfg.device = DEVICE
cfg.checkpoint_path = args.checkpoint or cfg.get("checkpoint_path")
cfg.compile = False
cfg.lr = args.lr
cfg.train_encoder = args.train_encoder
cfg._head_cfgs_flat = [dict(cfg)]
cfg._head_cfgs_flat[0]["name"] = "test_head"

model = build_multi_head_vjepa(cfg)
head_name = next(iter(model.heads.keys()))
head = model.heads[head_name]

head_cfg = EasyDict(dict(cfg))
head_cfg.device = DEVICE
head_cfg.name = head_name
criterion = build_loss(head_cfg)
opt, _ = build_optimizer(EasyDict({"lr": args.lr}), head, None)

model.to(DEVICE)
model.train()

# --- Synthetic data ---
B = 2
vd = torch.randn(B, 3, VCL, img_size, img_size, device=DEVICE)
di = torch.zeros(B, 11, device=DEVICE)
di[:, 0] = VCL
di[:, 1] = torch.arange(B)
di[:, 2] = -1
di[:, 3] = -1
di[:, 4] = -1
for b in range(B // 2, B):
    di[b, 2] = 4
    di[b, 3] = 6
    di[b, 4] = 1

print(f"Synthetic: {B} videos, {VCL} frames, labels={di[:, 4].int().tolist()}")
n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Trainable params in head: {n_params:,}")

# --- Feature diversity check ---
MAX_PAIRWISE = 2000  # cap tokens to avoid O(N²×D) broadcast OOM

def _sim_report(t, label):
    """Print pairwise cosine similarity stats for a set of vectors.
    Caps at MAX_PAIRWISE tokens (random subset) to stay within GPU memory."""
    flat = t.reshape(-1, t.shape[-1])
    n = flat.shape[0]
    if n > MAX_PAIRWISE:
        idx = torch.randperm(n, device=flat.device)[:MAX_PAIRWISE]
        flat = flat[idx]
        n_label = f"sampled {MAX_PAIRWISE}/{n}"
    else:
        n_label = str(n)
    # Chunked all-pairs cosine sim to avoid the [N,N,D] broadcast
    sims = []
    chunk = 128
    for i in range(0, flat.shape[0], chunk):
        q = flat[i : i + chunk]                              # [c, D]
        s = F.cosine_similarity(
            q.unsqueeze(1), flat.unsqueeze(0), dim=-1)       # [c, N]
        sims.append(s)
    sims = torch.cat(sims, dim=0)                            # [N, N]
    mask = ~torch.eye(sims.shape[0], dtype=torch.bool, device=sims.device)
    off = sims[mask]
    print(f"  {label}: tokens={n_label}  sim=[{off.min():.4f}, {off.max():.4f}]  mean={off.mean():.4f}")

with torch.no_grad():
    if args.train_encoder:
        patches_check = head.encoder(vd[:, :, :NF, :, :], return_patches=True)
        # patches_check: [B, N_patches, embed_dim]
        _sim_report(patches_check, "patch tokens")
        _sim_report(patches_check.mean(dim=1, keepdim=True), "pooled (mean)")
    else:
        patches_check = model.encode_video_clips(vd, NF)
        # patches_check: [B, n_clips, N_patches, embed_dim]
        _sim_report(patches_check, "patch tokens (all)")
        _sim_report(patches_check.mean(dim=2, keepdim=True), "pooled (mean over patches)")
        # Also check: are patches within one clip diverse, or all collapsed?
        if patches_check.shape[2] > 1:
            single_clip = patches_check[0, 0]  # [N_patches, embed_dim] — first clip
            _sim_report(single_clip, "patches within ONE clip")
    # Flatten to [total_tokens, D] for a summary similarity check.
    # Cap tokens to avoid the same O(N²×D) broadcast that _sim_report guards against
    # (9216 tokens × 768D × 4 bytes = 260 GiB broadcast intermediate).
    feats_flat = patches_check.reshape(-1, patches_check.shape[-1]) if head._needs_patches else patches_check.mean(dim=2).reshape(-1, patches_check.shape[-1])
    n_feats = feats_flat.shape[0]
    if n_feats > MAX_PAIRWISE:
        idx = torch.randperm(n_feats, device=feats_flat.device)[:MAX_PAIRWISE]
        feats_flat = feats_flat[idx]
    _sim_report(feats_flat.unsqueeze(0), "features (final)")

# --- Overfit ---
for epoch in range(args.epochs):
    print(epoch)
    if args.train_encoder:
        patches_in = vd
    else:
        with torch.no_grad():
            patches = model.encode_video_clips(vd, NF)
        patches_in = patches if head._needs_patches else patches.mean(dim=2)

    toa_batch = di[:, 2]
    tea_batch = di[:, 3]
    video_len_orig = di[:, 0]

    state = None
    epoch_loss = 0.0
    frame_count = 0

    for i in range(NF, VCL):
        target = gt_cls_target(i, toa_batch, tea_batch).long()

        if args.train_encoder:
            clip = vd[:, :, i - NF:i, :, :]
            feat = head.encoder(clip, return_patches=True)
            if not head._needs_patches:
                feat = feat.mean(dim=1)
        else:
            feat = patches_in[:, i - NF, ...]

        output, state = head.forward_temporal_step(feat, state)

        flt = i >= video_len_orig
        target[flt] = -100
        output[flt] = -100

        if apply_softmax:
            output = output.softmax(dim=1)

        loss = criterion(output, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.detach().item()
        frame_count += 1

    avg_loss = epoch_loss / max(frame_count, 1)

    if epoch == 0 or (epoch + 1) % 25 == 0:
        with torch.no_grad():
            if not args.train_encoder:
                patches_p = model.encode_video_clips(vd, NF)
                patches_in_p = patches_p if head._needs_patches else patches_p.mean(dim=2)
            state_p = None
            preds = []
            for i in range(NF, VCL):
                if args.train_encoder:
                    clip = vd[:, :, i - NF:i, :, :]
                    feat = head.encoder(clip, return_patches=True)
                    if not head._needs_patches:
                        feat = feat.mean(dim=1)
                else:
                    feat = patches_in_p[:, i - NF, ...]
                out, state_p = head.forward_temporal_step(feat, state_p)
                if cfg.get("apply_softmax", False):
                    out = out.softmax(dim=1)
                preds.append(out[:, 1].cpu())
            all_preds = torch.stack(preds, dim=1)

        # Check if params moved
        grads_ok = 0
        for p in head.parameters():
            if p.grad is not None and p.grad.norm().item() > 1e-10:
                grads_ok += 1
        print(f"  ep {epoch+1:3d}: loss={avg_loss:.6f}  pred=[{all_preds.min():.4f}, {all_preds.max():.4f}]  mean={all_preds.mean():.4f}  params_w_grad={grads_ok}")

final_loss = avg_loss
learned = abs(final_loss - 0.693) > 0.02 and final_loss < 0.65
status = "✓ LEARNED" if learned else "✗ STUCK"
print(f"\n=> {status}  final_loss={final_loss:.6f}")
