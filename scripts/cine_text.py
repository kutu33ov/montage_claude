#!/usr/bin/env python3
"""Cinematic text video from a photo (or a video): very light film look + soft word-by-word text.
Look: gentle S-curve with lifted blacks, teal shadows / warm highlights, soft bloom on lights,
slightly lower saturation, light vignette, fine moving grain. Photo gets a slow push-in.
Usage: cine_text.py config.json
config: {"src": "photo.jpg|video.mp4", "out": "out.mp4", "dur": 12, "fps": 60, "push": 0.06,
         "texts": [{"text": "...", "at": 0.5, "until": 5.0}, {"text": "...", "at": 5.5}],
         "y": 360, "size": 60, "look": 1.0}"""
import json, subprocess, sys, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FONT = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
c = json.load(open(sys.argv[1]))
if c.get('native') and c['src'].lower().endswith(('.mp4', '.mov', '.mkv', '.m4v')):   # keep the source resolution
    _wh = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                          '-of', 'csv=p=0', c['src']], capture_output=True, text=True).stdout.split(',')
    W, H = int(_wh[0]) // 2 * 2, int(_wh[1]) // 2 * 2
SC = W / 1080                                   # sizes in the config are given for 1080 wide
SHB, SHW = (14, 4) if c.get('shadow') == 'strong' else (7, 0)
SHB, SHW = SHB * W / 1080, int(SHW * W / 1080)   # "shadow": "strong" = clearly visible shadows
FPS, DUR = c.get('fps', 60), c.get('dur', 12.0)
N = int(FPS * DUR); K = c.get('look', 1.0)

def grade(rgb, k=K):
    """rgb float32 0..1 -> graded (subtle)"""
    L = rgb @ np.float32([0.2126, 0.7152, 0.0722])
    # bloom / halation on bright lights (warm)
    hi = np.clip((L - 0.72) / 0.28, 0, 1)[..., None] * rgb
    sm = cv2.resize(hi, (hi.shape[1] // 4, hi.shape[0] // 4), interpolation=cv2.INTER_AREA)
    bloom = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 22 * hi.shape[1] / 1080 / 4), (hi.shape[1], hi.shape[0])) * np.float32([1.0, 0.82, 0.62])
    x = rgb + bloom * c.get('bloom', 0.22) * k
    # gentle S-curve + lifted blacks, slightly soft whites
    x = np.clip(x, 0, 1)
    s = x + 0.06 * k * (x - 0.5) * (1 - np.abs(2 * x - 1))
    x = 0.025 * k + s * (1 - 0.045 * k)
    # split toning: teal shadows, warm highlights
    L = (x @ np.float32([0.2126, 0.7152, 0.0722]))[..., None]
    x = x + ((1 - L) ** 2) * np.float32([-0.012, 0.006, 0.02]) * k + (L ** 2) * np.float32([0.022, 0.008, -0.016]) * k
    # saturation down a touch
    L = (x @ np.float32([0.2126, 0.7152, 0.0722]))[..., None]
    x = L + (x - L) * (1 - 0.08 * k)
    # vignette
    yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1]].astype(np.float32)
    r = ((xx / x.shape[1] - 0.5) / 0.75) ** 2 + ((yy / x.shape[0] - 0.5) / 0.9) ** 2
    x = x * (1 - 0.2 * k * np.clip(r, 0, 1) ** 1.4)[..., None]
    return np.clip(x, 0, 1)

def text_frames(text, size, maxw=900):
    size = int(size * SC); maxw = int(maxw * SC)
    """layout words in centred lines; returns list of (word image RGBA, x, y)"""
    f = ImageFont.truetype(FONT, size); d = ImageDraw.Draw(Image.new('RGBA', (8, 8)))
    words, lines, cur = text.split(), [], []
    for w in words:
        t = ' '.join(cur + [w])
        if cur and d.textlength(t, font=f) > maxw: lines.append(cur); cur = [w]
        else: cur.append(w)
    lines.append(cur)
    lh = int(size * 1.32); sp = d.textlength(' ', font=f); out = []
    top = -lh * len(lines) / 2
    for i, ln in enumerate(lines):
        lw = d.textlength(' '.join(ln), font=f); x = (W - lw) / 2
        for w in ln:
            ww = int(d.textlength(w, font=f)) + 80
            im = Image.new('RGBA', (ww, lh + 80), (0, 0, 0, 0))
            sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
            ImageDraw.Draw(sh).text((40, 42), w, font=f, fill=(0, 0, 0, 255), stroke_width=SHW, stroke_fill=(0, 0, 0, 255))
            im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(SHB)))          # wide soft shadow
            if SHB > 8:                                                          # + tight contact shadow
                sh2 = Image.new('RGBA', im.size, (0, 0, 0, 0))
                ImageDraw.Draw(sh2).text((41, 43), w, font=f, fill=(0, 0, 0, 200))
                im.alpha_composite(sh2.filter(ImageFilter.GaussianBlur(2.5)))
            ImageDraw.Draw(im).text((40, 40), w, font=f, fill=(255, 250, 242, 255))
            out.append((im, x - 40, top + i * lh - 40))
            x += d.textlength(w, font=f) + sp
    return out

# source
is_video = c['src'].lower().endswith(('.mp4', '.mov', '.mkv', '.m4v'))
if not is_video:
    src = cv2.cvtColor(cv2.imread(c['src']), cv2.COLOR_BGR2RGB)
    k = max(W / src.shape[1], H / src.shape[0])
    big = cv2.resize(src, (int(src.shape[1] * k * 1.5), int(src.shape[0] * k * 1.5)), interpolation=cv2.INTER_LANCZOS4)
    base = grade(big.astype(np.float32) / 255)          # grade once at high res
    BH, BW = base.shape[:2]
else:
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', c['src'], '-vf',
                            f'fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)

texts = [dict(t, words=text_frames(t['text'], c.get('size', 60))) for t in c['texts']]
YC = c.get('y', 360) * SC
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        *(['-i', c['src'], '-map', '0:v', '-map', '1:a:0?'] if is_video else
                          ['-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo', '-map', '0:v', '-map', '1:a']),
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', str(c.get('crf', 16)), '-pix_fmt', 'yuv420p',
                        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                        '-c:a', 'aac', '-b:a', '192k', '-t', f'{DUR}', '-movflags', '+faststart', c['out']], stdin=subprocess.PIPE)
rng = np.random.default_rng(1)
for i in range(N):
    t = i / FPS
    if not is_video:
        z = 1 + c.get('push', 0.06) * (1 - (1 - t / DUR) ** 2)            # ease-out push-in
        sw = int(BW / z)
        sh = int(sw * H / W)
        sh = min(sh, BH); x0 = (BW - sw) // 2; y0 = int((BH - sh) * 0.3)
        fr = cv2.resize(base[y0:y0 + sh, x0:x0 + sw], (W, H), interpolation=cv2.INTER_AREA)
    else:
        b = dec.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        fr = grade(np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255)
    if c.get('grain', 0.016):                                                    # fine, pixel-level grain
        fr = np.clip(fr + rng.normal(0, c.get('grain', 0.016) * K, (H, W, 1)).astype(np.float32), 0, 1)
    img = Image.fromarray((fr * 255).astype(np.uint8)).convert('RGBA')
    for tx in texts:
        a0, a1 = tx['at'], tx.get('until', DUR + 1)
        if t < a0 or t > a1 + 0.7: continue
        out_a = 1 - float(np.clip((t - a1) / 0.7, 0, 1))
        for j, (im, x, y) in enumerate(tx['words']):
            p = float(np.clip((t - a0 - j * 0.11) / 0.55, 0, 1)); p = p * p * (3 - 2 * p)
            if p <= 0: continue
            al = p * out_a
            if al <= 0: continue
            w = im.copy(); w.putalpha(w.split()[3].point(lambda v: int(v * al)))
            if p < 1: w = w.filter(ImageFilter.GaussianBlur(float((1 - p) * 4 * SC)))
            img.alpha_composite(w, (int(x), int(YC + y + (1 - p) * 14 * SC)))
    enc.stdin.write(np.asarray(img.convert('RGB')).tobytes())
enc.stdin.close(); enc.wait()
print('ok', c['out'])
