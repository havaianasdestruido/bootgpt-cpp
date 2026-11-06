#!/usr/bin/env python3
"""Deterministic analysis of the committed quantized model (not a BIOS emulator).

Requires numpy. Prompts are left-padded with spaces and truncated to 32 chars;
the boot image itself has only its embedded seed, not a prompt interface.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import random
import subprocess
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sampling import sample, weights_for


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

    def generate(self, ctx, seed, count, trace=False, legacy=False):
        p = self.p
        ctx = list(ctx)
        rnd = seed & 0xffff
        result = []
        for position in range(count):
            a, h, logits = self.forward(ctx)
            if legacy:
                # Original decoder, retained solely for before/after analysis.
                threshold = int(logits.max()) - 240
                best, chosen = -(1 << 31), 0
                for i, logit in enumerate(logits):
                    rnd = (rnd * 25173 + 13849) & 0xffff
                    r = rnd >> 8
                    r = r - 256 if r >= 128 else r
                    candidate = int(logit) + r * 6
                    if logit >= threshold and candidate > best:
                        best, chosen = candidate, i
            else:
                chosen, rnd = sample(logits, rnd)
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
    parser.add_argument('--legacy', action='store_true', help='use original Min-P sampler')
    parser.add_argument('--corpus-file', type=Path, help='measure initial non-whitespace 6-gram corpus matches')
    parser.add_argument('--corpus-stats', action='store_true', help='compare softmax weights with 2000 corpus targets')
    parser.add_argument('--seed-sweep', action='store_true', help='evaluate boot seed with odd RNG seeds 1 to 39')
    parser.add_argument('--syntax-check', action='store_true', help='compile function-prompt completions with g++')
    args = parser.parse_args()
    model = Model()
    cases = [
        ('boot-seed-loop', model.p['seed_ctx'], 0x1235, None),
        ('short-completion', model.context('int add(int a, int b) { return '), 1,
         'int add(int a, int b) { return '),
        ('multiline-function', model.context('int sum(int n) {\n    int s = 0;\n'), 2,
         'int sum(int n) {\n    int s = 0;\n'),
        ('long-generation', model.p['seed_ctx'], 0xcaff, None),
        ('balanced-delimiters', model.context('int f(int x) {\n    if (x > 0) {\n'), 3,
         'int f(int x) {\n    if (x > 0) {\n'),
    ]
    corpus = args.corpus_file.read_text() if args.corpus_file else None
    if args.syntax_check and not shutil.which('g++'):
        parser.error('--syntax-check requires g++')
    if args.corpus_stats:
        if corpus is None:
            parser.error('--corpus-stats requires --corpus-file')
        generator = random.Random(123)
        distances, legacy_excluded, soft_zero = [], 0, 0
        for _ in range(2000):
            position = generator.randrange(model.p['C'], len(corpus))
            ctx = model.context(corpus[position - model.p['C']:position])
            target = model.vocab.index(corpus[position])
            logits = model.forward(ctx)[2]
            weights = np.array(weights_for(logits), dtype=np.float64)
            exact = np.exp((logits - logits.max()) / 256.0)
            distances.append(float(np.abs(weights / weights.sum() - exact / exact.sum()).sum() / 2))
            legacy_excluded += int(logits[target] < logits.max() - 240)
            soft_zero += int(weights[target] == 0)
        print(json.dumps({'case': 'corpus-sampler-stats', 'count': len(distances),
                          'softmax_tv_mean': float(np.mean(distances)),
                          'softmax_tv_max': max(distances),
                          'legacy_excluded_targets': legacy_excluded,
                          'softmax_zero_targets': soft_zero}))
    if args.seed_sweep:
        cases = [(f'boot-seed-{seed:04x}', model.p['seed_ctx'], seed, None)
                 for seed in range(1, 40, 2)]
    for name, context, seed, prompt in cases:
        text = model.generate(context, seed, args.tokens, args.trace, args.legacy)
        record = {'case': name, 'seed': seed, 'decoder': 'legacy' if args.legacy else 'epsilon-soft',
                  **metrics(text), 'output': text}
        if corpus is not None:
            window = text[:300]
            grams = [window[i:i + 6] for i in range(max(0, len(window) - 5))
                     if window[i:i + 6].strip()]
            record['initial_nonblank_6gram_match_fraction'] = (
                round(sum(g in corpus for g in grams) / len(grams), 4) if grams else None)
            record['initial_nonblank_6gram_count'] = len(grams)
        if args.syntax_check and prompt is not None:
            result = subprocess.run(['g++', '-std=c++20', '-x', 'c++', '-fsyntax-only', '-'],
                                    input=prompt + text, text=True, capture_output=True)
            record['syntax_ok'] = result.returncode == 0
            record['syntax_diagnostic'] = result.stderr.splitlines()[:2]
        print(json.dumps(record))


if __name__ == '__main__':
    main()
