#!/usr/bin/env python3
"""Train a bare-metal C++20 neural LM for the 512-byte boot sector,
quantize to int8/int16, and simulate the EXACT integer arithmetic in boot.asm.

Everything the boot sector does is defined here first; boot.asm must match.
"""
import json
from pathlib import Path

import numpy as np

from sampling import sample

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(7)

# ---------------- architecture (fixed dimensions, corpus-derived vocab) ----------------
C = 32      # context characters
E = 6       # embedding dim
H = 100     # hidden units
Q = 12      # activation fixed-point (Q12)
LQ = 8      # logit fixed-point (Q8, int16)

STEPS = 50000
BATCH = 512
LR = 0.01

# ---------------- data ----------------
text = open("corpus.txt").read()
freq = {}
for ch in text:
    freq[ch] = freq.get(ch, 0) + 1
top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
vocab = [ch for ch, _ in top]
V = len(vocab)
# IDs are bytes and the assembly reserves 512 bytes for int16 logits.
assert 0 < V <= 256, f"vocabulary of {V} characters exceeds boot buffers"
assert all(ch == "\n" or " " <= ch <= "~" for ch in vocab), \
    "corpus contains non-printable or non-ASCII characters"
# Check the disk/segment budget before the expensive training loop. The blob
# ends with a context and two NUL-terminated BIOS strings (see emission).
_header = (b"bootgpt-cpp: bare-metal neural network dreaming C++20"
           b"\r\nno os involved - press any key to reboot\r\n\r\n\x00")
_bye = b"\r\n// build interrupted - rebooting...\r\n\x00"
_syntax_size = (MODEL_DIR / "syntax.bin").stat().st_size
_prefix_size = (V + V * E + H * C * E + V * H + 2 * H + 2 * V + C
                + len(_header) + len(_bye))
_blob_size = _prefix_size + (-_prefix_size % 16) + _syntax_size
assert _blob_size <= 62 * 512 and 0x8000 + _blob_size < 0x10000, \
    f"vocabulary of {V} characters exceeds the boot weight budget"
assert "\n" in vocab and " " in vocab
vmap = {ch: i for i, ch in enumerate(vocab)}
space_id = vmap[" "]

ids = np.array([vmap[ch] for ch in text], dtype=np.int64)
print("corpus chars:", len(ids), "vocab size:", len(vocab), "vocab:", "".join(
    c if c != "\n" else "\\n" for c in vocab))

# Identifier-char indices for annealed distant-context dropout
ident_ids = np.array(
    [i for i, ch in enumerate(vocab) if ch.isalnum() or ch == "_"],
    dtype=np.int64,
)
is_ident = np.zeros(V, dtype=bool)
is_ident[ident_ids] = True

# Pick an exact in-distribution 32-char seed context ending at "\n\n" right
# before "#include <" so generated code starts cleanly at column 0 of line 1.
_seed_anchor = text.find("\n    return 0;\n}\n\n#pragma once\n\n#include <")
assert _seed_anchor >= 0, "seed anchor not found in corpus.txt"
SEED_STR = text[_seed_anchor : _seed_anchor + C]
assert len(SEED_STR) == C and SEED_STR.endswith("\n\n")
seed_ctx = [vmap[c] for c in SEED_STR]

# ---------------- model ----------------
emb = rng.normal(0, 0.12, (V, E))
W1 = rng.normal(0, 1.0 / np.sqrt(C * E), (H, C * E))   # [j][i]
b1 = np.zeros(H)
W2 = rng.normal(0, 1.0 / np.sqrt(H), (V, H))           # [v][h]
b2 = np.zeros(V)

params = [emb, W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def softclip(a):          # activation used by BOTH training and asm
    return a / (1.0 + np.abs(a))


def forward(x_idx):
    xin = emb[x_idx].reshape(len(x_idx), C * E)        # [B, C*E]
    a1 = xin @ W1.T + b1
    h = softclip(a1)
    logits = h @ W2.T + b2
    return xin, a1, h, logits


losses = []
DROPOUT_CUTOFF = int(STEPS * 0.65)
DISTANT_SPAN = C - 12     # keep the most recent 12 chars 100% clean

for step in range(1, STEPS + 1):
    pos = rng.integers(0, len(ids) - C - 1, BATCH)
    x_idx = np.stack([ids[pos + k] for k in range(C)], axis=1)   # [B, C]
    y = ids[pos + C]

    if step <= DROPOUT_CUTOFF:
        p_drop = 0.08 * (1.0 - (step / DROPOUT_CUTOFF) * 0.5)
        distant = x_idx[:, :DISTANT_SPAN]
        mask = (rng.random(distant.shape) < p_drop) & is_ident[distant]
        if mask.any():
            distant[mask] = rng.choice(ident_ids, size=int(mask.sum()))

    xin, a1, h, logits = forward(x_idx)
    p = softmax(logits)
    loss = -np.log(p[np.arange(BATCH), y] + 1e-9).mean()
    losses.append(loss)

    dlogits = p
    dlogits[np.arange(BATCH), y] -= 1
    dlogits /= BATCH

    gW2 = dlogits.T @ h
    gb2 = dlogits.sum(0)
    dh = dlogits @ W2
    da1 = dh * (1.0 / (1.0 + np.abs(a1)) ** 2)         # softclip grad
    gW1 = da1.T @ xin
    gb1 = da1.sum(0)
    gemb_flat = da1 @ W1                               # [B, C*E]
    gemb = np.zeros_like(emb)
    np.add.at(gemb, x_idx.reshape(-1), gemb_flat.reshape(-1, E))

    lr = LR * (0.04 ** (step / STEPS))
    for i, g in enumerate([gemb, gW1, gb1, gW2, gb2]):
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        mh = m[i] / (1 - beta1 ** step)
        vh = v[i] / (1 - beta2 ** step)
        params[i] -= lr * mh / (np.sqrt(vh) + eps)

    if step % 5000 == 0:
        print(f"step {step:6d}  loss {np.mean(losses[-5000:]):.4f}  lr {lr:.5f}")

emb, W1, b1, W2, b2 = params
np.savez(MODEL_DIR / "model.npz", emb=emb, W1=W1, b1=b1, W2=W2, b2=b2,
         vocab=np.array("".join(vocab)))


# ---------------- float baseline sampler ----------------
def gen_float(n, temp=0.8, seed0=None):
    ctx = list(seed0 if seed0 is not None else seed_ctx)
    out = []
    rs = np.random.default_rng(0)
    for _ in range(n):
        _, _, _, logits = forward(np.array([ctx]))
        pr = softmax(logits[0] / temp)
        nxt = rs.choice(V, p=pr)
        out.append(vocab[nxt])
        ctx = ctx[1:] + [nxt]
    return "".join(out)


print("\n--- FLOAT baseline sample ---")
print(gen_float(600))


# ---------------- quantization + QAT (Straight-Through Estimator) ----------------
def pct_pow2_scale(w, pct=99.8):
    """scale = 2^e based on 99.8th percentile so single outliers don't waste int8 bits."""
    mx = np.percentile(np.abs(w), pct)
    e = int(np.floor(np.log2(127.0 / mx))) if mx > 0 else 7
    while np.percentile(np.abs(w) * (1 << e), pct) > 127:
        e -= 1
    return e


def q_int8(w, e):
    return np.clip(np.round(w * (1 << e)), -127, 127).astype(np.int8)


def q_fix(w, q):
    return np.clip(np.round(w * (1 << q)), -32700, 32700).astype(np.int16)


# Rescale emb/W1 (function-preserving) to equalize relative quantization
# precision of both tensors, then select full-range power-of-two scales.
alpha = np.percentile(np.abs(emb), 99.5) / np.percentile(np.abs(W1), 99.5)
emb = emb * alpha
W1 = W1 / alpha

ee = pct_pow2_scale(emb)
ew1 = pct_pow2_scale(W1)
ew2 = pct_pow2_scale(W2)
A1SHIFT = ee + ew1 - Q
A2SHIFT = 6 + ew2 - LQ

# 5,000 steps of Quantization-Aware Training (QAT) with Straight-Through
# Estimator on the exact int8/int16 grid + integer softclip used by boot.asm.
QAT_STEPS = 5000
params_q = [emb, W1, b1, W2, b2]
mq = [np.zeros_like(p) for p in params_q]
vq = [np.zeros_like(p) for p in params_q]
q_scales = [1 << ee, 1 << ew1, 1 << Q, 1 << ew2, 1 << LQ]
q_clips = [127, 127, 32700, 127, 32700]
qat_losses = []

for qstep in range(1, QAT_STEPS + 1):
    pos = rng.integers(0, len(ids) - C - 1, BATCH)
    x_idx = np.stack([ids[pos + k] for k in range(C)], axis=1)
    y = ids[pos + C]

    emb_fq = np.clip(np.round(emb * (1 << ee)), -127, 127) / (1 << ee)
    W1_fq = np.clip(np.round(W1 * (1 << ew1)), -127, 127) / (1 << ew1)
    b1_fq = np.clip(np.round(b1 * (1 << Q)), -32700, 32700) / (1 << Q)
    W2_fq = np.clip(np.round(W2 * (1 << ew2)), -127, 127) / (1 << ew2)
    b2_fq = np.clip(np.round(b2 * (1 << LQ)), -32700, 32700) / (1 << LQ)

    xin = emb_fq[x_idx].reshape(BATCH, C * E)
    a1 = xin @ W1_fq.T + b1_fq
    h_smooth = softclip(a1)
    h_int = np.trunc(h_smooth * 64.0) / 64.0
    h = h_smooth + (h_int - h_smooth)      # STE integer softclip
    logits = h @ W2_fq.T + b2_fq

    p = softmax(logits)
    qat_losses.append(-np.log(p[np.arange(BATCH), y] + 1e-9).mean())

    dl = p
    dl[np.arange(BATCH), y] -= 1
    dl /= BATCH
    gW2 = dl.T @ h
    gb2 = dl.sum(0)
    dh = dl @ W2_fq
    da1 = dh * (1.0 / (1.0 + np.abs(a1)) ** 2)
    gW1 = da1.T @ xin
    gb1 = da1.sum(0)
    gemb_flat = da1 @ W1_fq
    gemb = np.zeros_like(emb)
    np.add.at(gemb, x_idx.reshape(-1), gemb_flat.reshape(-1, E))

    qlr = 0.0003 * (0.15 ** (qstep / QAT_STEPS))
    for i, g in enumerate([gemb, gW1, gb1, gW2, gb2]):
        mq[i] = beta1 * mq[i] + (1 - beta1) * g
        vq[i] = beta2 * vq[i] + (1 - beta2) * g * g
        mh = mq[i] / (1 - beta1 ** qstep)
        vh = vq[i] / (1 - beta2 ** qstep)
        params_q[i] -= qlr * mh / (np.sqrt(vh) + eps)
        lim = q_clips[i] / q_scales[i]
        np.clip(params_q[i], -lim, lim, out=params_q[i])

print(f"\nQAT fine-tune ({QAT_STEPS} steps): loss {np.mean(qat_losses[:500]):.4f} -> {np.mean(qat_losses[-500:]):.4f}")

emb_q = q_int8(emb, ee)
W1_q = q_int8(W1, ew1)        # stored [j][i]
W2_q = q_int8(W2, ew2)        # stored [v][h]
b1_q = q_fix(b1, Q)           # preact Q12
b2_q = q_fix(b2, LQ)          # logits Q8

nclip1 = int((np.abs(W1_q) == 127).sum() + (np.abs(emb_q) == 127).sum())
nclip2 = int((np.abs(W2_q) == 127).sum())
print(f"scales: emb 2^{ee}  w1 2^{ew1}  w2 2^{ew2}  "
      f"a1shift {A1SHIFT}  a2shift {A2SHIFT}  "
      f"sat(emb,W1) {nclip1} sat(W2) {nclip2}")

# ---------------- EXACT integer simulation of boot.asm ----------------
emb_f = emb_q.reshape(-1)                      # [v*E + r]
W1_f = W1_q.reshape(-1).astype(np.int64)       # [j*(C*E) + i]
W2_f = W2_q.reshape(-1).astype(np.int64)       # [v*H + h]
W1_m = W1_f.reshape(H, C * E)
W2_m = W2_f.reshape(V, H)
b1_i = b1_q.astype(np.int64)
b2_i = b2_q.astype(np.int64)


def trunc_div(a, b):          # x86 idiv semantics
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


tvec = np.vectorize(trunc_div)


class BootSim:
    """Mirrors boot.asm arithmetic instruction-for-instruction."""

    def __init__(self, seed=0x1234):
        self.rnd = seed & 0xFFFF

    def preacts_and_logits(self, ctx):
        xin = np.array([emb_f[ctx[k] * E + r]
                        for k in range(C) for r in range(E)],
                       dtype=np.int64)
        a1 = W1_m @ xin
        p = (a1 >> A1SHIFT) if A1SHIFT >= 0 else (a1 << -A1SHIFT)
        p = p + b1_i
        y = tvec(p * 64, 4096 + np.abs(p))
        y = np.clip(y, -64, 64)
        a2 = W2_m @ y
        logits = ((a2 >> A2SHIFT) if A2SHIFT >= 0
                  else (a2 << -A2SHIFT)) + b2_i
        return p, y, logits

    def logits_for(self, ctx):
        return self.preacts_and_logits(ctx)[2]

    def step_token(self, ctx):
        token, self.rnd = sample(self.logits_for(ctx), self.rnd)
        return token

    def gen(self, n, ctx0):
        ctx = list(ctx0)
        out = []
        for _ in range(n):
            bi = self.step_token(ctx)
            out.append(vocab[bi])
            ctx = ctx[1:] + [bi]
        return "".join(out)


# Overflow guard: preacts*64 must fit int32 and logits must fit int16 (Q8)
_probe = BootSim(seed=0x1234)
_pmax, _lmax = 0, 0
_ctx = list(seed_ctx)
for _ in range(800):
    _p, _, _l = _probe.preacts_and_logits(_ctx)
    _pmax = max(_pmax, int(np.abs(_p).max()))
    _lmax = max(_lmax, int(np.abs(_l).max()))
    _ctx = _ctx[1:] + [_probe.step_token(_ctx)]
print(f"max |p| on sample walk: {_pmax} (int32/64 limit 33554431)")
print(f"max |logit| on sample walk: {_lmax} (int16 limit 32767)")
assert _pmax < 10_000_000, _pmax
assert _lmax < 30_000, _lmax

print("\n--- INTEGER (asm-identical) epsilon-soft samples across seeds ---")
for sd in [0x1234, 0xBEEF, 0xCAFE]:
    s = BootSim(seed=sd)
    txt = s.gen(900, seed_ctx)
    print(f"\n[seed={sd:#06x}, epsilon-soft Q14]")
    print(txt)

# ---------------- emit artifacts ----------------
blob = bytearray()
blob += bytes(ord(c) if c != "\n" else 10 for c in vocab)   # V
blob += emb_q.tobytes()                                     # V*E
blob += W1_q.tobytes()                                      # H*C*E
blob += W2_q.tobytes()                                      # V*H
blob += b1_q.tobytes()                                      # H*2
blob += b2_q.tobytes()                                      # V*2
expected = V + V * E + H * C * E + V * H + 2 * H + 2 * V
assert len(blob) == expected, (len(blob), expected)

# Seed context and strings live on disk too; the boot sector is pure code.
SEED_ADDR = 0x8000 + len(blob)
blob += bytes(seed_ctx)

HDR_ADDR = 0x8000 + len(blob)
blob += _header
BYE_ADDR = 0x8000 + len(blob)
blob += _bye

# The syntax highlighter is position-independent code loaded with the model.
# Keeping it here leaves the MBR's scarce 510 bytes available for inference.
blob += b"\x00" * (-len(blob) % 16)
SYNTAX_ADDR = 0x8000 + len(blob)
with (MODEL_DIR / "syntax.bin").open("rb") as syntax_file:
    blob += syntax_file.read()
assert 0x8000 + len(blob) < 0x10000, "DS=0 16-bit address overflow"

SECTORS = (len(blob) + 511) // 512
assert SECTORS <= 62, f"weights ({SECTORS} sectors) exceed single CHS track read"
blob += b"\x00" * (SECTORS * 512 - len(blob))
(MODEL_DIR / "weights.bin").write_bytes(blob)

with (MODEL_DIR / "params.inc").open("w") as f:
    w1off = V + V * E
    w2off = w1off + H * C * E
    b1off = w2off + V * H
    b2off = b1off + 2 * H
    f.write(f"; auto-generated by train.py\n"
            f"VOCAB    equ {V}\n"
            f"CTX      equ {C}\n"
            f"EMBDIM   equ {E}\n"
            f"HID      equ {H}\n"
            f"W1_OFF   equ {w1off}\n"
            f"W2_OFF   equ {w2off}\n"
            f"B1_OFF   equ {b1off}\n"
            f"B2_OFF   equ {b2off}\n"
            f"A1SHIFT  equ {A1SHIFT}\n"
            f"A2SHIFT  equ {A2SHIFT}\n"
            f"SECTORS  equ {SECTORS}\n"
            f"SEEDADDR equ {SEED_ADDR:#x}\n"
            f"HDRADDR  equ {HDR_ADDR:#x}\n"
            f"BYEADDR  equ {BYE_ADDR:#x}\n"
            f"SYNTAXADDR equ {SYNTAX_ADDR:#x}\n")

with (MODEL_DIR / "params.json").open("w") as f:
    json.dump({
        "V": V, "C": C, "E": E, "H": H,
        "A1SHIFT": A1SHIFT, "A2SHIFT": A2SHIFT,
        "SAMPLER": "epsilon-soft-q14",
        "SECTORS": SECTORS,
        "scales": [ee, ew1, ew2],
        "seed_ctx": seed_ctx,
    }, f, indent=2)
print(f"\nmodel/weights.bin: {len(blob)} bytes ({SECTORS} sectors), "
      "model/params.inc written")
