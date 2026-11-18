#!/usr/bin/env python3
"""Synthesise a royalty-free starter pack: sound effects into assets/sfx and a placeholder
music loop into assets/music. Everything is generated from scratch (no third-party samples).
Usage: python3 scripts/gen_sfx.py"""
import os, wave
import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
rng = np.random.default_rng(42)

def save(path, x, stereo=None):
    x = np.asarray(x, np.float64)
    if stereo is None: stereo = np.stack([x, x], 1) if x.ndim == 1 else x
    stereo = stereo / max(1e-9, np.abs(stereo).max()) * 0.89
    data = (stereo * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(data.tobytes())
    print('wrote', os.path.relpath(path, ROOT))

def t(d): return np.arange(int(SR * d)) / SR
def env(n, a, r, curve=3.0):
    e = np.ones(n); na = max(1, int(SR * a)); nr = max(1, int(SR * r))
    e[:na] = np.linspace(0, 1, na) ** 2
    e[-nr:] *= np.linspace(1, 0, nr) ** curve
    return e
def bp(x, lo, hi, order=2): return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)
def lp(x, f, order=2): return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x)
def hp(x, f, order=2): return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)

def sweep_noise(d, f0, f1, width=0.6, block=512):
    """noise through a band-pass whose centre glides f0 -> f1 (overlap-add blocks)"""
    n = int(SR * d); noise = rng.standard_normal(n + block); out = np.zeros(n + block)
    win = np.hanning(2 * block)
    for i in range(0, n, block):
        fc = f0 * (f1 / f0) ** (i / n)
        seg = noise[max(0, i - block):i + block]
        y = bp(seg, max(40, fc * (1 - width / 2)), min(SR / 2 - 100, fc * (1 + width / 2)))
        y = y[-2 * block:] if len(y) >= 2 * block else np.pad(y, (2 * block - len(y), 0))
        s = max(0, i - block); out[s:s + 2 * block] += (y * win)[:len(out[s:s + 2 * block])]
    return out[:n]

def whoosh(d=0.55, f0=300, f1=4500):
    x = sweep_noise(d, f0, f1); n = len(x)
    e = np.sin(np.linspace(0, np.pi, n)) ** 1.6
    x = x * e
    pan = np.linspace(-0.8, 0.8, n)
    return np.stack([x * np.sqrt((1 - pan) / 2), x * np.sqrt((1 + pan) / 2)], 1)

def pop(d=0.12):
    tt = t(d); f = 900 * np.exp(-tt * 30) + 280
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-tt * 38)
    x[:120] += rng.standard_normal(120) * np.linspace(1, 0, 120) * 0.5
    return x

def click(d=0.05):
    tt = t(d)
    return hp(rng.standard_normal(len(tt)), 2500) * np.exp(-tt * 260) + np.sin(2 * np.pi * 3200 * tt) * np.exp(-tt * 180) * 0.5

def ding(d=1.4, f=1318.5):
    tt = t(d)
    x = sum(a * np.sin(2 * np.pi * f * k * tt) * np.exp(-tt * dec) for k, a, dec in
            ((1, 1, 3.2), (2.76, .45, 5.5), (5.4, .25, 9), (8.93, .12, 14)))
    return x * env(len(tt), 0.002, 0.3)

def riser(d=1.6):
    tt = t(d)
    nz = sweep_noise(d, 400, 7000, 0.5)
    f = 180 * (8 ** (tt / d)); tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.35
    e = (tt / d) ** 2.2
    return (nz * 0.8 + tone) * e * env(len(tt), 0.01, 0.04)

def boom(d=1.2):
    tt = t(d); f = 95 * np.exp(-tt * 3) + 38
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 3.5)
    thump = lp(rng.standard_normal(len(tt)), 300) * np.exp(-tt * 18) * 2
    return np.tanh((sub + thump) * 1.6)

def glitch(d=0.35):
    n = int(SR * d); x = np.zeros(n); i = 0
    while i < n:
        L = int(rng.uniform(0.008, 0.04) * SR); f = rng.choice([180, 360, 720, 1440, 2880])
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(L) / SR)) * rng.uniform(0.3, 1)
        if rng.random() < 0.4: seg = rng.standard_normal(L) * 0.6
        x[i:i + L] = seg[:n - i]; i += L + int(rng.uniform(0, 0.01) * SR)
    x = np.round(x * 6) / 6
    return x * env(n, 0.002, 0.05)

def swipe(d=0.25): return whoosh(d, 1200, 7000) * 0.8

def music_loop(bars=8, bpm=88):
    """soft lo-fi bed: pad chords + plucked arpeggio + gentle kick/hat. Loops seamlessly."""
    beat = 60 / bpm; bar = 4 * beat; d = bars * bar; n = int(SR * d); tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    # Am - F - C - G (x2)
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]] * (bars // 4)
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    for b, ch in enumerate(chords):
        s, e = int(b * bar * SR), int((b + 1) * bar * SR); L = e - s; tl = np.arange(L) / SR
        pad = sum(np.sin(2 * np.pi * hz(m) * tl + 0.3 * np.sin(2 * np.pi * 0.3 * tl)) +
                  0.3 * np.sin(2 * np.pi * hz(m + 12) * tl * 1.002) for m in ch)
        pad = lp(pad, 1400) * env(L, 0.35, 0.45, 1.5) * 0.18
        out[s:e, 0] += pad; out[s:e, 1] += pad * 0.95
        arp = ch + [ch[1] + 12]
        for k in range(8):  # eighth notes
            ps = s + int(k * beat / 2 * SR); pl = int(beat / 2 * SR * 1.6)
            pl = min(pl, n - ps); tp = np.arange(pl) / SR
            note = np.sin(2 * np.pi * hz(arp[k % 4] + 12) * tp) * np.exp(-tp * 7) * 0.12
            pan = 0.35 if k % 2 else -0.35
            out[ps:ps + pl, 0] += note * (1 - pan) / 2 * 2; out[ps:ps + pl, 1] += note * (1 + pan) / 2 * 2
        bass = np.sin(2 * np.pi * hz(ch[0] - 12) * tl) * env(L, 0.02, 0.3, 2) * 0.22
        out[s:e] += bass[:, None]
    for k in range(int(d / beat)):  # kick on 1 & 3, hat on offbeats
        p = int(k * beat * SR)
        if k % 2 == 0:
            tk = np.arange(int(0.25 * SR)) / SR
            kick = np.sin(2 * np.pi * np.cumsum(110 * np.exp(-tk * 25) + 45) / SR) * np.exp(-tk * 12) * 0.35
            L = min(len(kick), n - p); out[p:p + L] += kick[:L, None]
        hpos = p + int(beat / 2 * SR); L = min(int(0.05 * SR), n - hpos)
        if L > 0:
            hat = hp(rng.standard_normal(L), 7000) * np.exp(-np.arange(L) / SR * 90) * 0.05
            out[hpos:hpos + L] += hat[:, None]
    # tiny crossfade at loop point
    xf = int(0.02 * SR); out[:xf] *= np.linspace(0, 1, xf)[:, None]; out[-xf:] *= np.linspace(1, 0, xf)[:, None]
    return out

if __name__ == '__main__':
    sd = os.path.join(ROOT, 'assets', 'sfx'); md = os.path.join(ROOT, 'assets', 'music')
    os.makedirs(sd, exist_ok=True); os.makedirs(md, exist_ok=True)
    save(f'{sd}/whoosh.wav', None, whoosh())
    save(f'{sd}/swipe.wav', None, swipe())
    save(f'{sd}/pop.wav', pop())
    save(f'{sd}/click.wav', click())
    save(f'{sd}/ding.wav', ding())
    save(f'{sd}/riser.wav', riser())
    save(f'{sd}/boom.wav', boom())
    save(f'{sd}/glitch.wav', glitch())
    save(f'{md}/placeholder_lofi_88bpm.wav', None, music_loop())
