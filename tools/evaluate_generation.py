#!/usr/bin/env python3
"""Deterministic analysis of the committed quantized model (not a BIOS emulator).

Requires numpy. Prompts are left-padded with spaces and truncated to 32 chars;
the boot image itself has only its embedded seed, not a prompt interface.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Model:
    def __init__(self, directory=ROOT / 'model'):
        p = json.loads((directory / 'params.json').read_text())
        self.p = p
        data = (directory / 'weights.bin').read_bytes()
        v, c, e, h = (p[k] for k in ('V', 'C', 'E', 'H'))
        self.vocab = data[:v].decode('ascii')
        off = v
        def take(shape, dtype):
            nonlocal off
            dtype = np.dtype(dtype)
            count = int(np.prod(shape))
            value = np.frombuffer(data, dtype=dtype, count=count, offset=off)
            off += count * dtype.itemsize
            return value.reshape(shape).astype(np.int64)
        self.emb = take((v, e), 'i1')
        self.w1 = take((h, c * e), 'i1')
        self.w2 = take((v, h), 'i1')
        self.b1 = take((h,), '<i2')
        self.b2 = take((v,), '<i2')

    def context(self, text):
        c = self.p['C']
        text = text[-c:].rjust(c)
        return [self.vocab.index(ch) for ch in text]

    def forward(self, ctx):
        p = self.p
        a = self.w1 @ self.emb[ctx].reshape(-1)
        shift = p['A1SHIFT']
        a = (a >> shift if shift >= 0 else a << -shift) + self.b1
        h = np.sign(a) * ((np.abs(a) * 64) // (4096 + np.abs(a)))
        l = self.w2 @ h
        shift = p['A2SHIFT']
        l = (l >> shift if shift >= 0 else l << -shift) + self.b2
        return a, h, l

    def generate(self, ctx, seed, count, trace=False):
        p = self.p
        ctx = list(ctx)
        rnd = seed & 0xffff
        result = []
        for position in range(count):
            a, h, logits = self.forward(ctx)
            threshold = int(logits.max()) - p['MARGIN']
            best, chosen = -(1 << 31), 0
            for i, logit in enumerate(logits):
                rnd = (rnd * 25173 + 13849) & 0xffff
                r = rnd >> 8
                r = r - 256 if r >= 128 else r
                candidate = int(logit) + r * p['NOISE']
                if logit >= threshold and candidate > best:
                    best, chosen = candidate, i
            if trace:
                print(json.dumps({'position': position, 'context_ids': ctx,
                                  'token_id': chosen, 'token': self.vocab[chosen],
                                  'logit_min': int(logits.min()),
                                  'logit_max': int(logits.max()),
                                  'logit_selected': int(logits[chosen]),
                                  'preact_abs_max': int(np.abs(a).max()),
                                  'hidden_abs_max': int(np.abs(h).max())}))
            result.append(self.vocab[chosen])
            ctx = ctx[1:] + [chosen]
        return ''.join(result)


def metrics(text):
    grams = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
    return {'chars': len(text),
            'repeated_4gram_fraction': round(1 - len(set(grams)) / len(grams), 4) if grams else 0,
            'longest_space_run': max(map(len, re.findall(' +', text)), default=0),
            'newlines': text.count('\n')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=1500)
    parser.add_argument('--trace', action='store_true', help='emit per-token JSON')
    args = parser.parse_args()
    model = Model()
    cases = [
        ('boot-seed-loop', model.p['seed_ctx'], 0x1235),
        ('short-completion', model.context('int add(int a, int b) { return '), 1),
        ('multiline-function', model.context('int sum(int n) {\n    int s = 0;\n'), 2),
        ('long-generation', model.p['seed_ctx'], 0xcaff),
        ('balanced-delimiters', model.context('int f(int x) {\n    if (x > 0) {\n'), 3),
    ]
    for name, context, seed in cases:
        text = model.generate(context, seed, args.tokens, args.trace)
        print(json.dumps({'case': name, 'seed': seed, **metrics(text), 'output': text}))


if __name__ == '__main__':
    main()
