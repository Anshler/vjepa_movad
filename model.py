"""
MOVAD anomaly classifier with a frozen V-JEPA 2.1 encoder backbone.

Supports four temporal model variants:
  - ``lstm``           — 3-layer LSTM (original MOVAD design)
  - ``mamba``          — 3 Mamba SSM blocks (mamba_ssm package)
  - ``slotssm``        — modular slots, per-slot Mamba, cross+self-attention
  - ``sparse_slotssm`` — SlotSSM + top-k sparse gating

SlotSSM architecture follows the reference repo (NeurIPS 2024) 1:1:
  block = Norm→CrossAttn(inverted,multi-head)→+res → Norm→Mamba→+res → Norm→SelfAttn→+res

All variants target ~15–25M trainable parameters with a frozen V-JEPA encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vjepa_encoder import VJEPA2Encoder, build_vjepa2_encoder

# ---------------------------------------------------------------------------
# Mamba_ssm import
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba, Mamba2

    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    Mamba = None
    Mamba2 = None


def _require_mamba():
    if not _HAS_MAMBA_SSM:
        raise ImportError(
            "mamba_ssm is required for this temporal model. "
            "Install with: pip install mamba-ssm causal-conv1d"
        )


# ---------------------------------------------------------------------------
# Flash-attn import (optional — speeds up SlotSSM self/cross attention).
# Requires a flash-attn wheel built against the EXACT PyTorch + CUDA version.
# Pre-built wheels: https://github.com/Dao-AILab/flash-attention/releases
# ---------------------------------------------------------------------------
try:
    from flash_attn.modules.mha import MHA as FlashMHA

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    FlashMHA = None


# ---------------------------------------------------------------------------
# MambaCache — identical to the SlotSSM repo.
# ---------------------------------------------------------------------------
@dataclass
class MambaCache:
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MultiHeadAttention — from the SlotSSM reference repo (NeurIPS 2024).
#
# Supports *inverted* attention where softmax runs over the slot dimension
# instead of the feature dimension, forcing input features to compete for
# slot assignment.  This encourages slot specialization without auxiliary
# losses.  The reference repo uses this as the default (train.py:120).
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional inverted softmax.

    Standard:  softmax over source (features) — each slot picks features.
    Inverted:  softmax over target (slots) — features compete for slots.
    """

    def __init__(
        self, d_model, num_heads, dropout=0.0, inverted=False, bias=True,
        norm_over_input=True, epsilon=1e-5,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input
        self.epsilon = epsilon

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = nn.Linear(d_model, d_model, bias=bias)
        self.proj_k = nn.Linear(d_model, d_model, bias=bias)
        self.proj_v = nn.Linear(d_model, d_model, bias=bias)
        self.proj_o = nn.Linear(d_model, d_model, bias=bias)

        # Populated during forward when inverted=True (see diagnostics in forward)
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

    def forward(self, q, k, v):
        B, T, _ = q.shape
        _, S, _ = k.shape

        q_proj = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k_proj = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v_proj = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q_proj = q_proj * (q_proj.shape[-1] ** (-0.5))
        attn = torch.matmul(q_proj, k_proj.transpose(-1, -2))

        if self.inverted:
            # Softmax over (head * target) → features compete over slots
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(
                B, self.num_heads, T, S,
            )
            # --- diagnostics: per-slot mass BEFORE re-normalization ----------
            # Raw softmax mass per slot (summed over heads and patches).
            # Each of S patches distributes 1.0 across h×T entries by inverted
            # softmax, so fair share = S/T.  Normalize to fraction of fair share:
            # 1.0 = exactly fair share, < 0.05 = at risk, < 1e-4 = dead.
            pre_norm_mass = attn.detach().sum(dim=(1, -1))           # [B, T]
            fair = S / T                                               # fair-share mass per slot
            mass_norm = pre_norm_mass / fair                           # [B, T], fair share = 1.0
            self._slot_mass_min = mass_norm.min()                      # worst slot fraction
            self._slot_mass_mean = mass_norm.mean()                    # avg across slots
            # Fraction of slots receiving at least 15% of fair share
            self._slot_usage_frac = (mass_norm > 0.15).float().mean()
            # -----------------------------------------------------------------
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v_proj).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_frozen(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)


def _print_param_summary(encoder, temporal, classifier, extra_parts=None):
    frozen = _count_frozen(encoder)
    trainable = _count(temporal) + _count(classifier)
    if extra_parts:
        for _tag, mod in extra_parts:
            trainable += _count(mod) if mod is not None else 0

    print(f"  Frozen (encoder):     {frozen / 1e6:.1f}M")
    print(f"  Trainable (temporal): {_count(temporal) / 1e6:.2f}M")
    print(f"  Trainable (classifier+proj): {_count(classifier) / 1e6:.2f}M")
    if extra_parts:
        for tag, mod in extra_parts:
            n = _count(mod) if mod is not None else 0
            print(f"  Trainable ({tag}): {n / 1e6:.2f}M")
    print("  ---")
    print(f"  Total trainable:      {trainable / 1e6:.2f}M")
    print(f"  Total frozen:         {frozen / 1e6:.1f}M")


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


# ===========================================================================
# Temporal model: identity (no temporal modelling — pure per-frame MLP probe)
# ===========================================================================
class NoTemporalModel(nn.Module):
    """Identity passthrough that returns no state.

    Used for diagnostic linear probing — replaces any recurrent/SSM model so
    each frame is classified independently.  If features carry discriminative
    signal, even this simple per-frame MLP will beat random.
    """

    def forward(self, x, state=None):
        return x, None


# ===========================================================================
# Temporal model: LSTM
# ===========================================================================
class LSTMTemporalModel(nn.Module):
    def __init__(self, dim: int, hidden_size: int, num_layers: int = 3):
        super().__init__()
        self.rnn = nn.LSTM(dim, hidden_size, num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        x = self.norm(x).unsqueeze(0)
        x, new_state = self.rnn(x, state)
        x = x.squeeze(0)
        hx, cx = new_state
        return x, (hx.detach(), cx.detach())


# ===========================================================================
# Temporal model: Mamba (streaming via MambaCache)
# ===========================================================================
class MambaTemporalModel(nn.Module):
    def __init__(
        self, dim: int, expand: int = 2, d_state: int = 128, d_conv: int = 4,
        num_blocks: int = 3, mamba_version: str = "mamba2",
    ):
        super().__init__()
        _require_mamba()
        mamba_cls = Mamba2 if mamba_version == "mamba2" else Mamba
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_blocks):
            kw = dict(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i)
            if mamba_version == "mamba2":
                kw["headdim"] = 64
                assert (dim * expand / 64) % 8 == 0, (
                    f"Mamba2 requires (d_model * expand / headdim) %% 8 == 0, "
                    f"got ({dim} * {expand} / 64) = {dim * expand / 64}"
                )
            self.blocks.append(mamba_cls(**kw))
            self.norms.append(nn.LayerNorm(dim))

    def forward(self, x, cache: MambaCache | None = None):
        if cache is None:
            cache = MambaCache()
        h = x.unsqueeze(1)
        for blk, norm in zip(self.blocks, self.norms):
            h = blk(norm(h), inference_params=cache) + h
        cache.seqlen_offset += 1
        return h.squeeze(1), cache


# ===========================================================================
# SlotSSM — follows the reference repo (NeurIPS 2024) block-for-block.
#
#   ref (raw V-JEPA patches)  →  each block projects independently
#   slots                     →  cross-attn (inverted, multi-head)
#                             →  +res → Mamba(per-slot) → +res
#                             →  +res → SelfAttn(slots) → +res
# ===========================================================================


class SlotSSMBlock(nn.Module):
    """
    One SlotSSM block — matches the reference repo 1:1.

        ref → input_proj → ref_norm
        slots → slot_norm ─┤
        cross-attn(standard or inverted, multi-head) → +residual
        → [sparse gate: top-k mask]
        → Mamba(per-slot, streaming)     → +residual
        → SelfAttn(across slots)         → +residual

    Dense (``top_k=None``): all K slots update every step (reference behaviour).

    Sparse (``top_k=int``): only top-k slots are *active* per timestep.
    Inactive slots are truly frozen — no cross-attn update, no Mamba, no
    self-attn update.  Their representation is preserved bit-for-bit across
    steps, serving as long-term memory.  Active slots can still *read* from
    inactive slots via self-attention (inactive slots are KV-only).

    Cross-attention mode
    --------------------
    ``use_inverted_attention=False`` (default):
        Standard FlashMHA cross-attention.  Each slot independently picks
        which features to attend to via softmax over the feature dimension.
        Multiple slots can attend to the same features — no competition.

    ``use_inverted_attention=True``:
        Inverted softmax (from the SlotSSM reference repo, module.py).
        Softmax runs over (head × slot) dimensions, so each feature token
        competes to be claimed by a slot.  Single-head by default to
        encourage object-level segmentation (reference repo, train.py:122).
        Uses eager MultiHeadAttention — FlashMHA doesn't support inverted.
    """

    def __init__(
        self, slot_dim: int, input_dim: int, top_k: int | None = None,
        mamba_d_state: int = 128, mamba_d_conv: int = 4, mamba_expand: int = 2,
        mamba_version: str = "mamba2", num_heads: int = 4, block_idx: int = 0,
        eps_random: float = 0.0,
        use_inverted_attention: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.top_k = top_k
        self.eps_random = eps_random
        self.use_inverted_attention = use_inverted_attention
        mamba_cls = Mamba2 if mamba_version == "mamba2" else Mamba

        self.input_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Cross-attention
        #  - inverted:   always eager MultiHeadAttention (FlashMHA incompatible)
        #  - standard:  FlashMHA when available, else nn.MultiheadAttention
        self.cross_attn_input_norm = nn.LayerNorm(slot_dim)
        self.cross_attn_ref_norm = nn.LayerNorm(slot_dim)
        if use_inverted_attention:
            # Single head matches the reference repo default (train.py:122):
            #   encoder_attn_num_heads=1  # for inverted attn to encourage object segmentation
            self.cross_attn = MultiHeadAttention(
                d_model=slot_dim, num_heads=num_heads, inverted=True,
            )
            self._cross_attn_inverted = True
        elif _HAS_FLASH_ATTN:
            self.cross_attn = FlashMHA(embed_dim=slot_dim, num_heads=num_heads, cross_attn=True)
            self._cross_attn_inverted = False
        else:
            self.cross_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
            self._cross_attn_inverted = False

        # Diagnostics populated during forward (only meaningful for inverted path)
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

        # Sparse gate — only when top_k is set
        if top_k is not None:
            self.gate = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, 1, bias=False),
            )
        self._gate_entropy = torch.tensor(0.0)  # accumulated per forward pass

        # Per-slot Mamba
        kw = dict(d_model=slot_dim, d_state=mamba_d_state, d_conv=mamba_d_conv,
                  expand=mamba_expand, layer_idx=block_idx)
        if mamba_version == "mamba2":
            kw["headdim"] = 64
            # Reference asserts this constraint (slotssm.py:153)
            assert (slot_dim * mamba_expand / kw["headdim"]) % 8 == 0, (
                f"Mamba2 requires (d_model * expand / headdim) %% 8 == 0, "
                f"got ({slot_dim} * {mamba_expand} / {kw['headdim']}) = "
                f"{slot_dim * mamba_expand / kw['headdim']}"
            )
        self.mamba = mamba_cls(**kw)
        self.time_mixer_norm = nn.LayerNorm(slot_dim)

        # Slot self-attention
        # - Dense: self-attn (Q=KV=slots), FlashMHA when available.
        # - Sparse: active slots query, all slots as KV (Q≠KV).
        #   FlashMHA self-attn can't do this, so sparse always uses eager
        #   nn.MultiheadAttention.  At 32×32 with batch=1 it's tiny (0.47ms
        #   vs 0.21ms FlashMHA) — not worth a dedicated FlashMHA cross-attn
        #   instance (4.2M params across 4 blocks).
        self.space_attn_norm = nn.LayerNorm(slot_dim)
        if _HAS_FLASH_ATTN and top_k is None:
            self.self_attn = FlashMHA(embed_dim=slot_dim, num_heads=num_heads)
        else:
            self.self_attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cross_attn(self, slots, ref_raw):
        ref_proj = self.input_proj(ref_raw)                       # [B, N, D]
        q = self.cross_attn_input_norm(slots)                     # [B, K, D]
        kv = self.cross_attn_ref_norm(ref_proj)                   # [B, N, D]
        if self._cross_attn_inverted:
            out = self.cross_attn(q, kv, kv)
            # Forward diagnostics from the inverted MultiHeadAttention
            self._slot_mass_min = self.cross_attn._slot_mass_min
            self._slot_mass_mean = self.cross_attn._slot_mass_mean
            self._slot_usage_frac = self.cross_attn._slot_usage_frac
        elif _HAS_FLASH_ATTN:
            out = self.cross_attn(x=q, x_kv=kv)
        else:
            out = self.cross_attn(q, kv, kv)[0]
        return out

    def _mamba_step(self, slots, cache):
        """Run Mamba on all slots. Dense path only."""
        x = self.time_mixer_norm(slots).reshape(-1, 1, slots.shape[-1])
        return self.mamba(x, inference_params=cache).reshape_as(slots)

    def _mamba_step_sparse(self, slots, active_flat, cache):
        """Run Mamba only on active slots.

        First call (cache not yet initialised for this block): runs the full
        scan path on all slots to initialise the Mamba cache.  Matches the
        prior behaviour where inactive slots advanced once before being frozen
        from step 2 onward.

        Subsequent calls: extracts active-slot states from the cache as
        contiguous tensors, calls Mamba2.step() directly on the active subset,
        then scatters the updated states and outputs back.  Inactive slots are
        never touched — no clone, no restore, ~50 % less Mamba compute.
        """
        B, K, D = slots.shape
        layer_idx = self.mamba.layer_idx
        x = self.time_mixer_norm(slots).reshape(-1, 1, D)

        has_prev = layer_idx in cache.key_value_memory_dict
        if not has_prev:
            # First call: scan path initialises the cache for all slots and
            # advances every state by one timestep (same as the original
            # save/restore path, which also had no previous state to restore).
            full_out = self.mamba(x, inference_params=cache).reshape(B, K, D)
            return full_out   # caller applies the mask

        # --- Subsequent calls: active-slot-only step ------------------------
        kv = cache.key_value_memory_dict[layer_idx]

        # Extract active slots as contiguous tensors (required by the fused
        # CUDA / Triton kernels inside Mamba2.step).
        conv_active = kv[0][active_flat].contiguous()          # [n_active, C, d_conv]
        ssm_active = kv[1][active_flat].contiguous()           # [n_active, H, hdim, d_state]
        x_active = x[active_flat]                              # [n_active, 1, D]

        # step() mutates conv_active & ssm_active in-place.
        out_active, _, _ = self.mamba.step(x_active, conv_active, ssm_active)

        # Scatter the updated states back into the cache.
        kv[0][active_flat] = conv_active
        kv[1][active_flat] = ssm_active

        # Scatter the output into a zeroed tensor (inactive slots → 0).
        out = torch.zeros(B * K, D, device=slots.device, dtype=out_active.dtype)
        out[active_flat] = out_active.squeeze(1)               # remove seqlen dim
        return out.reshape(B, K, D)

    def _self_attn_all(self, slots):
        x = self.space_attn_norm(slots)
        result = self.self_attn(x) if _HAS_FLASH_ATTN else self.self_attn(x, x, x)[0]
        return result

    def _self_attn_sparse(self, slots, active_flat):
        """Active slots query; all slots serve as KV (read-only memory).

        Always uses eager nn.MultiheadAttention — FlashMHA self-attn can't
        do Q≠KV, and the 16×32 attn at batch=1 is below FlashMHA's breakeven.
        """
        B, K, D = slots.shape
        x = self.space_attn_norm(slots)                        # [B, K, D]
        kv = x                                                  # [B, K, D]
        x_flat = x.reshape(B * K, D)                            # [B*K, D]
        q = x_flat[active_flat].reshape(B, -1, D)              # [B, active_slots, D]

        out_active = self.self_attn(q, kv, kv)[0]

        out = torch.zeros(B, K, D, device=slots.device, dtype=out_active.dtype)
        out_flat = out.reshape(B * K, D)
        out_flat[active_flat] = out_active.reshape(-1, D)
        return out

    # ------------------------------------------------------------------
    def forward(self, slots, ref_raw, cache: MambaCache):
        if self.top_k is None:
            # === Dense: all slots update (reference SlotSSM) ===
            slots = slots + self._cross_attn(slots, ref_raw)
            slots = slots + self._mamba_step(slots, cache)
            slots = slots + self._self_attn_all(slots)
            return slots

        # === Sparse: only top-k active slots update ===
        B, K, D = slots.shape
        layer_idx = self.mamba.layer_idx
        has_prev = layer_idx in cache.key_value_memory_dict

        # 1. Cross-attn for all K slots — needed so the gate can see the
        #    current input.  Each slot queries the scene from its frozen
        #    state; the resulting cross_out encodes "what this slot sees."
        cross_out = self._cross_attn(slots, ref_raw)

        # 2. Gate on input-informed representation (RIMs-style: the input
        #    drives activation, not just pre-existing slot state).
        informed = slots + cross_out
        gate_scores = self.gate(informed).squeeze(-1)                # [B, K]

        if self.training and self.eps_random > 0 and torch.rand(1).item() < self.eps_random:
            active_idx = torch.stack([torch.randperm(K, device=slots.device)[:self.top_k]
                                       for _ in range(B)])
            self._gate_entropy = gate_scores.new_tensor(float(math.log(K)))
        else:
            _, active_idx = gate_scores.topk(self.top_k, dim=1)      # [B, top_k]
            p = gate_scores.softmax(dim=-1)                           # [B, K]
            entropy = -(p * (p + 1e-9).log()).sum(dim=-1).mean()     # scalar
            self._gate_entropy = entropy.detach()

        # --- First step: Mamba cache not yet initialised for this block.
        #     Run scan path on all K slots to populate the cache, then
        #     fall through to the mask-based path (same as before).
        if not has_prev:
            mask = torch.zeros(B, K, device=slots.device, dtype=slots.dtype).scatter_(1, active_idx, 1.0)
            mask_3d = mask.unsqueeze(-1)
            active_flat = mask.reshape(-1).bool()
            slots = slots + cross_out * mask_3d
            x_all = self.time_mixer_norm(slots).reshape(-1, 1, D)
            full_out = self.mamba(x_all, inference_params=cache).reshape(B, K, D)
            slots = slots + full_out * mask_3d
            slots = slots + self._self_attn_sparse(slots, active_flat)
            return slots

        # === Subsequent steps: compact active-slot path -------------------
        # Operate on a dense [B, top_k, D] tensor using integer advanced
        # indexing, which produces contiguous views.  This eliminates all
        # mask multiplications and gather/scatter in attention, keeping the
        # CUDA graph fused and avoiding the ~1 ms/block sync overhead.

        tk = self.top_k
        batch_idx = torch.arange(B, device=slots.device).unsqueeze(1)   # [B, 1]

        # 3. Compact active slots — integer indexing → contiguous
        compact_slots = slots[batch_idx, active_idx]                     # [B, tk, D]
        compact_cross = cross_out[batch_idx, active_idx]                 # [B, tk, D]

        # 4. Cross-attn update (dense on compact, no mask)
        compact_slots = compact_slots + compact_cross

        # 5. Mamba on compact active slots
        x_compact = self.time_mixer_norm(compact_slots).reshape(-1, 1, D)  # [B*tk, 1, D]
        idx_flat = active_idx.reshape(-1)                                   # [B*tk]
        kv = cache.key_value_memory_dict[layer_idx]
        conv_active = kv[0][idx_flat]       # [B*tk, C, d_conv] — integer idx → contiguous
        ssm_active = kv[1][idx_flat]        # [B*tk, H, hdim, d_state]
        out_active, _, _ = self.mamba.step(x_compact, conv_active, ssm_active)
        kv[0][idx_flat] = conv_active       # scatter updated states back
        kv[1][idx_flat] = ssm_active
        compact_slots = compact_slots + out_active.squeeze(1).reshape(B, tk, D)

        # 6. Self-attn: compact Q queries full slots as KV (read-only memory)
        compact_q = self.space_attn_norm(compact_slots)                # [B, tk, D]
        full_kv = self.space_attn_norm(slots)                           # [B, K, D]
        sa_out = self.self_attn(compact_q, full_kv, full_kv)[0]        # [B, tk, D]
        compact_slots = compact_slots + sa_out

        # 7. Scatter compact back into full slots (only active indices change).
        #     Clone to avoid in-place on a leaf-variable view.
        slots = slots.clone()
        slots[batch_idx, active_idx] = compact_slots
        return slots


class SlotSSMTemporalModel(nn.Module):
    """
    SlotSSM: K modular slots with independent Mamba dynamics.

    When ``top_k`` is None → dense (all slots update every step).
    When ``top_k`` is int  → sparse (only top-k active; inactive slots freeze).

    Follows the reference repo: initial slots are learnable, ref (raw V-JEPA
    patches) is passed to every block, each block projects independently.
    """

    def __init__(
        self, num_slots: int = 32, slot_dim: int = 512, input_dim: int = 1408,
        num_blocks: int = 4, top_k: int | None = None,
        mamba_d_state: int = 128, mamba_d_conv: int = 4, mamba_expand: int = 2,
        mamba_version: str = "mamba2", num_heads: int = 4,
        eps_random: float = 0.0,
        use_inverted_attention: bool = False,
    ):
        super().__init__()
        _require_mamba()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.top_k = top_k

        self.slots_init = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SlotSSMBlock(
                slot_dim=slot_dim, input_dim=input_dim, top_k=top_k,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, mamba_version=mamba_version,
                num_heads=num_heads, block_idx=i,
                eps_random=eps_random if top_k is not None else 0.0,
                use_inverted_attention=use_inverted_attention,
            )
            for i in range(num_blocks)
        ])
        self._entropy = torch.tensor(0.0)  # populated during forward
        self._slot_mass_min = torch.tensor(float("nan"))
        self._slot_mass_mean = torch.tensor(float("nan"))
        self._slot_usage_frac = torch.tensor(float("nan"))

    def forward(self, patches, cache: MambaCache | None = None):
        B = patches.shape[0]
        if cache is None:
            cache = MambaCache()

        slots = self.slots_init.expand(B, -1, -1)
        ent = 0.0
        for blk in self.blocks:
            slots = blk(slots, patches, cache)
            if blk.top_k is not None:
                ent = ent + blk._gate_entropy
        self._entropy = ent  # training loop reads this

        # Aggregate inverted cross-attn diagnostics across blocks (worst-case)
        self._slot_mass_min = min(blk._slot_mass_min for blk in self.blocks)
        self._slot_mass_mean = (sum(blk._slot_mass_mean for blk in self.blocks) / len(self.blocks))
        self._slot_usage_frac = min(blk._slot_usage_frac for blk in self.blocks)

        cache.seqlen_offset += 1
        return slots, cache                                          # [B, K, D]


# ===========================================================================
# Main model
# ===========================================================================
class ClsVJEPA(nn.Module):
    """
    V-JEPA 2.1 encoder → (pool or patches) → temporal model → binary classifier.
    """

    def __init__(
        self, encoder: VJEPA2Encoder, embed_dim: int,
        dim_latent: int = 1024, dropout: float = 0.5, temporal_model: str = "lstm",
        # LSTM
        rnn_state_size: int = 1024, rnn_cell_num: int = 3,
        # Mamba / SSM
        mamba_d_state: int = 128, mamba_d_conv: int = 4,
        mamba_expand: int = 2, mamba_version: str = "mamba2",
        # SlotSSM
        num_slots: int = 32, slot_dim: int = 512, num_ssm_blocks: int = 4,
        # Sparse SlotSSM
        top_k: int = 16,
        eps_random: float = 0.0,
        # Inverted attention (SlotSSM reference repo style)
        use_inverted_attention: bool = False,
        train_encoder: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_type = temporal_model
        self.train_encoder = train_encoder

        # ---- Slot-based path ------------------------------------------------
        if temporal_model in ("slotssm", "sparse_slotssm"):
            _require_mamba()
            self._slot_based = True
            is_sparse = temporal_model == "sparse_slotssm"
            self.temporal = SlotSSMTemporalModel(
                num_slots=num_slots, slot_dim=slot_dim, input_dim=embed_dim,
                num_blocks=num_ssm_blocks, top_k=top_k if is_sparse else None,
                mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand, mamba_version=mamba_version,
                eps_random=eps_random if is_sparse else 0.0,
                use_inverted_attention=use_inverted_attention,
            )

            # Learned attention-pool over slots (640 params — negligible).
            # A learnable query attends to slots via dot-product, letting the
            # classifier upweight anomaly-relevant slots instead of blending
            # all 32 equally.  Then a post-temporal MLP mirroring the standard
            # path's lin2 ensures both architectures have the same depth.
            self.slot_query = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
            D = slot_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, dim_latent),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_latent, dim_latent),   # mirrors lin2
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_latent, 2),
            )
            self.apply(_weights_init)

            mode = "sparse" if is_sparse else "dense"
            if verbose:
                print(f"\n[ClsVJEPA] SlotSSM ({mode}) — Parameter summary:")
                _print_param_summary(encoder, self.temporal, self.classifier)
            return

        # ---- Standard path --------------------------------------------------
        self._slot_based = False
        self.lin1 = nn.Linear(embed_dim, dim_latent)
        self.lin2 = nn.Linear(dim_latent, dim_latent)
        self.lin3 = nn.Linear(dim_latent, 2)
        self.bn = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

        if temporal_model == "lstm":
            self.temporal = LSTMTemporalModel(dim_latent, rnn_state_size, rnn_cell_num)
        elif temporal_model == "mamba":
            _require_mamba()
            self.temporal = MambaTemporalModel(
                dim=dim_latent, expand=mamba_expand,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                num_blocks=rnn_cell_num, mamba_version=mamba_version,
            )
        elif temporal_model == "none":
            self.temporal = NoTemporalModel()
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}")

        self.apply(_weights_init)

        if verbose:
            print("\n[ClsVJEPA] Parameter summary:")
            _print_param_summary(
                encoder, self.temporal,
                nn.ModuleList([self.lin1, self.lin2, self.lin3, self.bn]),
            )

    def encode_video_clips(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Run the frozen ViT over every NF-frame sliding window in one batch of videos.

        Stacks all stride-1 clips into a mega-batch and encodes in a single
        ``no_grad`` pass.  Memory scales with ``batch_size × VCL`` — keep VCL
        reasonable to avoid OOM.

        Returns ``[B, n_clips, N_patches, embed_dim]`` where ``n_clips = T - num_frames``.
        """
        B, C, T, H, W = x.shape
        n_clips = T - num_frames

        clips = []
        for i in range(num_frames, T):
            clips.append(x[:, :, i - num_frames:i, :, :])              # [B, C, NF, H, W]  (view)
        mega_batch = torch.cat(clips, dim=0)                            # [B * n_clips, C, NF, H, W]

        # N_patches = T_tokens × H_patches × W_patches  (deterministic from config)
        enc = self.encoder.encoder
        N_patches = (num_frames // enc.tubelet_size) * (H // enc.patch_size) * (W // enc.patch_size)

        # When training the encoder, let gradients flow; otherwise freeze
        ctx = torch.no_grad() if not self.train_encoder else torch.enable_grad()
        with ctx:
            patches = self.encoder(mega_batch, return_patches=True)     # [B * n_clips, N_patches, embed_dim]

        patches = patches.view(B, n_clips, N_patches, -1)               # [B, n_clips, N_patches, embed_dim]
        return patches

    def forward_temporal_step(
        self, x: torch.Tensor, state: torch.Tensor | tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple | None]:
        """Single temporal step from pre-computed encoder output.

        Args:
            x:     For standard path: ``[B, embed_dim]`` (spatially mean-pooled).
                   For slot path: ``[B, N_patches, embed_dim]`` (full patch tokens).
            state: temporal-model state (MambaCache or LSTM tuple).

        Returns:
            output:    ``[B, 2]`` class logits.
            new_state: updated temporal-model state.
        """
        if self._slot_based:
            slots, new_state = self.temporal(x, state)
            D = slots.shape[-1]
            scores = (slots * self.slot_query).sum(dim=-1) / (D ** 0.5)
            attn = scores.softmax(dim=-1)
            pooled = (attn.unsqueeze(-1) * slots).sum(dim=1)
            return self.classifier(pooled), new_state

        # Standard path: bn → lin1 → relu → temporal → lin2 → relu → lin3
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x, new_state = self.temporal(x, state)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)
        return x, new_state

    def forward(self, x, state=None):
        if self._slot_based:
            patches = self.encoder(x, return_patches=True)    # [B, N, embed_dim]
            slots, new_state = self.temporal(patches, state)  # [B, K, D]
            # Learned attention-pool: query attends to slots
            D = slots.shape[-1]
            scores = (slots * self.slot_query).sum(dim=-1) / (D ** 0.5)  # [B, K]
            attn = scores.softmax(dim=-1)                       # [B, K]
            pooled = (attn.unsqueeze(-1) * slots).sum(dim=1)   # [B, D]
            return self.classifier(pooled), new_state

        x = self.encoder(x)                                   # [B, embed_dim]
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x, new_state = self.temporal(x, state)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)
        return x, new_state


# ===========================================================================
# Multi-Head Wrapper — shared frozen encoder, multiple independent temporal
# heads trained on the same encoded features from a single ViT pass.
# ===========================================================================
class MultiHeadVJEPA(nn.Module):
    """One V-JEPA encoder → multiple temporal + classifier heads.

    Each head trains independently (its own optimizer, checkpoint, and
    TensorBoard writer).  The encoder runs *once* per batch and the resulting
    patch features are reused by all heads.  Losses are never summed — each
    head's ``.backward()`` flows only through its own parameters.

    When ``train_encoder=True`` the encoder is unfrozen and trained jointly.
    This mode assumes a **single head** — the encoder gradients come from one
    temporal model only.

    Usage
    -----
    >>> model = build_multi_head_vjepa(cfg)   # cfg.temporal_heads = [...]
    >>> patches = model.encode_video_clips(video_data, fb)   # encode once
    >>>
    >>> # Train head-by-head
    >>> for name, head in model.heads.items():
    >>>     opt = optimizers[name]
    >>>     opt.zero_grad()
    >>>     feat = patches if head._slot_based else patches.mean(dim=2)
    >>>     loss = run_temporal_loop(head, feat, ...)
    >>>     loss.backward()
    >>>     opt.step()
    """

    def __init__(self, encoder: VJEPA2Encoder, heads_configs: list[dict],
                 train_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.train_encoder = train_encoder

        if self.train_encoder:
            if len(heads_configs) > 1:
                raise ValueError(
                    f"train_encoder=True only supports a single head, got {len(heads_configs)}. "
                    "Multiple heads would produce conflicting encoder gradients."
                )
            # load_pretrained_encoder() froze these — reverse it
            for p in self.encoder.parameters():
                p.requires_grad = True

        self.heads = nn.ModuleDict()
        self.head_configs: dict[str, dict] = {}

        for head_cfg in heads_configs:
            name = head_cfg["name"]
            if name in self.heads:
                raise ValueError(f"Duplicate head name: {name}")
            self.head_configs[name] = dict(head_cfg)

            self.heads[name] = ClsVJEPA(
                encoder=encoder,
                embed_dim=encoder.embed_dim,
                dim_latent=head_cfg.get("dim_latent", 1024),
                dropout=head_cfg.get("dropout", 0.5),
                temporal_model=head_cfg["temporal_model"],
                rnn_state_size=head_cfg.get("rnn_state_size", 1024),
                rnn_cell_num=head_cfg.get("rnn_cell_num", 3),
                mamba_d_state=head_cfg.get("mamba_d_state", 128),
                mamba_d_conv=head_cfg.get("mamba_d_conv", 4),
                mamba_expand=head_cfg.get("mamba_expand", 2),
                mamba_version=head_cfg.get("mamba_version", "mamba2"),
                num_slots=head_cfg.get("num_slots", 32),
                slot_dim=head_cfg.get("slot_dim", 512),
                num_ssm_blocks=head_cfg.get("num_ssm_blocks", 4),
                top_k=head_cfg.get("top_k", 16),
                eps_random=head_cfg.get("eps_random", 0.0),
                use_inverted_attention=head_cfg.get("use_inverted_attention", False),
                verbose=False,
                train_encoder=train_encoder,
            )

        # Summarise
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        total_trainable = 0
        enc_label = "Trainable" if self.train_encoder else "Frozen"
        print(f"\n[MultiHeadVJEPA] {len(self.heads)} heads — shared encoder, independent temporal models")
        print(f"  {enc_label} (encoder): {enc_params / 1e6:.1f}M")
        for name, head in self.heads.items():
            tp = _count(head)
            total_trainable += tp
            print(f"  Head '{name}' ({head.temporal_type}): {tp / 1e6:.2f}M trainable")
        print("  ---")
        print(f"  Total trainable (all heads): {total_trainable / 1e6:.2f}M")

    def train(self, mode: bool = True):
        """Set training mode — temporal heads follow ``mode``, encoder stays eval
        unless ``train_encoder=True``."""
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    def encode_video_clips(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Run the ViT encoder once over all stride-1 clips in one batch.

        Delegates to the first head's :meth:`ClsVJEPA.encode_video_clips`
        (all heads share the same encoder, so the output is identical).
        Gradients flow through the encoder when ``train_encoder=True``.

        Returns ``[B, n_clips, N_patches, embed_dim]`` — raw patch tokens.
        Standard-path heads call ``.mean(dim=2)`` on this; slot-based heads
        use the full tensor.
        """
        return next(iter(self.heads.values())).encode_video_clips(x, num_frames)


# ===========================================================================
# Factories
# ===========================================================================
def build_cls_vjepa(cfg) -> ClsVJEPA:
    encoder = build_vjepa2_encoder(cfg)

    if cfg.get("compile", True) and hasattr(torch, "compile"):
        # max-autotune-no-cudagraphs preserves operator fusion + kernel
        # autotuning but avoids CUDA graphs, which pre-allocate every
        # intermediate across the captured region.  With CUDA graphs, a
        # 696-clip mega-batch through ViT-Base would need ~240 GB VRAM
        # (12 layers × ~20 GB/layer of pre-allocated intermediates).
        encoder.encoder = torch.compile(
            encoder.encoder, mode="max-autotune-no-cudagraphs",
        )

    model = ClsVJEPA(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        dim_latent=cfg.get("dim_latent", 1024),
        dropout=cfg.get("dropout", 0.5),
        temporal_model=cfg.get("temporal_model", "lstm"),
        rnn_state_size=cfg.get("rnn_state_size", 1024),
        rnn_cell_num=cfg.get("rnn_cell_num", 3),
        mamba_d_state=cfg.get("mamba_d_state", 128),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        mamba_expand=cfg.get("mamba_expand", 2),
        mamba_version=cfg.get("mamba_version", "mamba2"),
        num_slots=cfg.get("num_slots", 32),
        slot_dim=cfg.get("slot_dim", 512),
        num_ssm_blocks=cfg.get("num_ssm_blocks", 4),
        top_k=cfg.get("top_k", 16),
        eps_random=cfg.get("eps_random", 0.0),
        use_inverted_attention=cfg.get("use_inverted_attention", False),
    ).to(cfg.device)

    return model


def build_multi_head_vjepa(cfg) -> MultiHeadVJEPA:
    """Build a MultiHeadVJEPA from a multi-config CLI invocation.

    The CLI produces ``cfg._head_cfgs_flat`` — a list of dicts, each the
    full parsed YAML from one ``--config`` path, with a ``"name"`` field
    derived from the file basename.  The first config is the master (encoder,
    data, training settings); each subsequent config contributes its
    ``temporal_model`` settings.

    Every head inherits shared defaults from the master config (``dim_latent``,
    ``dropout``, etc.) but can override them per-head.
    """
    encoder = build_vjepa2_encoder(cfg)

    if cfg.get("compile", True) and hasattr(torch, "compile"):
        encoder.encoder = torch.compile(
            encoder.encoder, mode="max-autotune-no-cudagraphs",
        )

    head_configs = []
    for hc in cfg._head_cfgs_flat:
        merged = dict(hc)   # full YAML from that config file
        head_configs.append(merged)

    model = MultiHeadVJEPA(
        encoder=encoder, heads_configs=head_configs,
        train_encoder=cfg.get("train_encoder", False),
    ).to(cfg.device)
    return model
