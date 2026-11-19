#!/usr/bin/env python3
"""Face-locked stabilisation for vertical walking/selfie footage (e.g. 4K 2160x3840).
Pass 1: YuNet face centre on a small copy of every frame. Pass 2: the frame is scaled to `margin`x the
output size and a 1080x1920 window follows the (lightly smoothed) face, so the face stays put and the
walking shake disappears. Output: high-quality 1080x1920 intermediate with the original audio.
Usage: stabilize.py in.mp4 out.mp4 [--margin 1.22] [--fps 30]"""
import sys, subprocess, json, os
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin

src, dst = sys.argv[1], sys.argv[2]
def arg(k, d): return float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
M, FPS = arg('--margin', 1.22), int(arg('--fps', 30))
OW, OH = 1080, 1920
BW, BH = int(OW * M) // 2 * 2, int(OH * M) // 2 * 2
SW, SH = 270, 480

def frames(w, h):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'fps={FPS},scale={w}:{h}:flags=bicubic',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()

# pass 1: face track
pts = []
for f in frames(SW, SH):
    b, lm = skin.yunet_detect(f)
    pts.append(None if b is None else ((lm[0] + lm[1]) / 2 + lm[2]) / 2)   # between eyes and nose tip
n = len(pts)
idx = [i for i, p in enumerate(pts) if p is not None]
xy = np.array([pts[i] for i in idx]) * (BW / SW)
cx = np.interp(np.arange(n), idx, xy[:, 0]); cy = np.interp(np.arange(n), idx, xy[:, 1])
k = cv2.getGaussianKernel(int(FPS * 0.4) | 1, FPS * 0.07).ravel()               # kill detector jitter only
pad = len(k) // 2
cx = np.convolve(np.pad(cx, pad, mode='edge'), k, 'valid'); cy = np.convolve(np.pad(cy, pad, mode='edge'), k, 'valid')
# where should the face sit in the output: its median position in the plain (centre-cropped) framing
tx = np.median(cx) - (BW - OW) / 2; ty = np.median(cy) - (BH - OH) / 2
x0 = np.clip(cx - tx, 0, BW - OW).astype(int); y0 = np.clip(cy - ty, 0, BH - OH).astype(int)
json.dump({'fps': FPS, 'n': n, 'detected': len(idx)}, open(dst + '.stab.json', 'w'))

# pass 2: crop + encode (visually lossless intermediate), copy audio
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
                        '-i', src, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', '-shortest', dst], stdin=subprocess.PIPE)
for i, f in enumerate(frames(BW, BH)):
    j = min(i, n - 1)
    enc.stdin.write(np.ascontiguousarray(f[y0[j]:y0[j] + OH, x0[j]:x0[j] + OW]).tobytes())
enc.stdin.close(); enc.wait()
print(f'stabilised {n} frames, face found in {len(idx)}, window moved x {x0.min()}..{x0.max()}, y {y0.min()}..{y0.max()}')
