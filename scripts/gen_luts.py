#!/usr/bin/env python3
"""Generate a few .cube colour LUTs into assets/luts (use with "lut": "warm" in config)."""
import os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
N = 33

def curve(x, k):  # soft S-curve, k = strength
    return np.clip(x + k * (x - 0.5) * (1 - np.abs(2 * x - 1)), 0, 1)

def write(name, fn):
    g = np.linspace(0, 1, N)
    lines = [f'TITLE "{name}"', f'LUT_3D_SIZE {N}']
    for b in g:
        for gg in g:
            for r in g:
                o = np.clip(fn(np.array([r, gg, b])), 0, 1)
                lines.append(f'{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}')
    open(os.path.join(ROOT, 'assets', 'luts', f'{name}.cube'), 'w').write('\n'.join(lines) + '\n')
    print('wrote', name)

def sat(c, s):
    l = c @ np.array([0.299, 0.587, 0.114]); return l + (c - l) * s

write('warm', lambda c: sat(curve(c, 0.15) * [1.04, 1.0, 0.93] + [0.01, 0.005, 0], 1.08))
write('cool', lambda c: sat(curve(c, 0.15) * [0.95, 1.0, 1.05], 1.02))
write('contrast', lambda c: sat(curve(c, 0.35), 1.12))
write('film', lambda c: sat(curve(c, 0.2) * 0.92 + 0.05 + np.array([0.01, 0, -0.015]) * (1 - c), 0.88))
