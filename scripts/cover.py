#!/usr/bin/env python3
"""9:16 cover: natural retouched frame (soft bottom fade) + sentence-case title bottom-left,
key word on a violet brush stroke. Optional violet grade with "grade": true (owner did NOT like it)."""
import json, subprocess, sys, os, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skin import detect_face, smooth_skin, fix_nose

W, H = 1080, 1920
FONT = '/usr/share/fonts/opentype/inter/Inter-SemiBold.otf'

def grab(src, t, cx, zoom, face_y=0.33):
    bx = int(min(max(cx - 304, 0), 1920 - 608))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', src, '-frames:v', '1', '-vf', f'crop=608:1080:{bx}:0',
                          '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    b = detect_face(im)
    if b is None: b = np.array([164, 260, 280, 280])
    im = fix_nose(smooth_skin(im, b, 0.5, 1, 0.6), b)
    cw, ch = int(608 / zoom), int(1080 / zoom)
    fx, fy = b[0] + b[2] / 2, b[1] + b[3] / 2
    x0 = int(min(max(fx - cw / 2, 0), 608 - cw)); y0 = int(min(max(fy - ch * face_y, 0), 1080 - ch))
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LANCZOS4)

def grade(bgr, tint):
    """Duotone-ish grade: luminance mapped through shadow/mid/highlight colours, blended with a bit of original."""
    img = bgr[..., ::-1].astype(np.float32) / 255
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    lum = np.clip((lum - 0.06) / 0.86, 0, 1)
    lum = lum ** 0.95
    lum = lum * lum * (3 - 2 * lum) * 0.85 + lum * 0.15                       # gentle S-curve
    sh, md, hi = [np.array(c, np.float32) / 255 for c in (tint['shadow'], tint['mid'], tint['high'])]
    l = lum[..., None]
    lo = sh * (1 - l / 0.5) + md * (l / 0.5)
    up = md * (1 - (l - 0.5) / 0.5) + hi * ((l - 0.5) / 0.5)
    duo = np.where(l < 0.5, lo, up)
    k = tint.get('keep', 0.22)
    gray = (img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    sat = np.clip(gray + (img - gray) * 1.35, 0, 1)            # keep skin warmth, a bit richer
    out = duo * (1 - k) + sat * k
    # bloom
    br = np.clip(out - 0.62, 0, 1)
    out = out + cv2.GaussianBlur(br, (0, 0), 25) * 0.55
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = ((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H * 0.4) / (H * 0.62)) ** 2
    out = out * (1 - 0.5 * np.clip(v, 0, 1) ** 1.3)[..., None]
    # bottom fade for the title
    g = np.clip((yy - H * 0.55) / (H * 0.45), 0, 1) ** 1.2
    out = out * (1 - 0.55 * g)[..., None] + np.array(sh)[None, None] * (0.55 * g)[..., None] * 0.6
    # fine grain
    rng = np.random.default_rng(7)
    out = out + rng.normal(0, 0.018, out.shape[:2])[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

def brush(w, h, color, seed=1):
    """Skewed brush-stroke highlight with ragged edges."""
    rnd = random.Random(seed)
    pad = 30
    im = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    sk = h * 0.18
    top, bot = [], []
    n = 14
    for i in range(n + 1):
        x = pad + w * i / n
        top.append((x + sk + rnd.uniform(-4, 4), pad + rnd.uniform(-5, 6)))
        bot.append((x - sk + rnd.uniform(-4, 4), pad + h + rnd.uniform(-6, 5)))
    left = [(pad - rnd.uniform(4, 18), pad + h * k / 4 + rnd.uniform(-3, 3)) for k in range(4, -1, -1)]
    d.polygon(top + bot[::-1] + left, fill=color)
    # dry-brush streaks on the ends
    for _ in range(10):
        y = pad + rnd.uniform(0.1, 0.9) * h
        x1 = pad + w + rnd.uniform(-10, 14); x0 = x1 - rnd.uniform(30, 90)
        d.line((x0 - sk * (y - pad - h / 2) / h, y, x1, y), fill=color, width=int(rnd.uniform(3, 8)))
    return im.filter(ImageFilter.GaussianBlur(0.7))

def plain(bgr):
    """Natural frame, only a soft neutral fade at the bottom so the title reads."""
    out = bgr[..., ::-1].astype(np.float32) / 255
    yy = np.arange(H, dtype=np.float32)[:, None]
    g = np.clip((yy - H * 0.58) / (H * 0.42), 0, 1) ** 1.3
    out = out * (1 - 0.6 * g)[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))

def make(c):
    tint = c.get('tint', {'shadow': (18, 8, 42), 'mid': (110, 62, 190), 'high': (238, 214, 255)})
    frame = grab(c['src'], c['t'], c['cx'], c.get('zoom', 1.25), c.get('face_y', 0.33))
    img = (grade(frame, tint) if c.get('grade', False) else plain(frame)).convert('RGBA')
    size = c.get('size', 112)
    f = ImageFont.truetype(FONT, size)
    d = ImageDraw.Draw(img)
    x0 = c.get('x', 84)
    lines = c['lines']
    # shrink to fit width
    def lw(ln, f): return sum(d.textlength(s, font=f) for s, _ in ln) + d.textlength(' ', font=f) * (len(ln) - 1)
    while max(lw(ln, f) for ln in lines) > W - 2 * x0 - 30 and size > 60:
        size -= 4; f = ImageFont.truetype(FONT, size)
    lh = int(size * 1.2)
    y = c.get('y', 1560) - lh * len(lines)
    accent = tuple(c.get('accent', (124, 58, 255)))
    # soft text shadow layer
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(shadow)
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    yy = y
    for li, ln in enumerate(lines):
        x = x0
        for si, (seg, hl) in enumerate(ln):
            tw = d.textlength(seg, font=f)
            if hl:
                b = brush(int(tw + 44), int(size * 0.98), accent + (255,), seed=li * 7 + si)
                layer.alpha_composite(b, (int(x - 22 - 30), int(yy + size * 0.12 - 30)))
            sd.text((x, yy + 4), seg, font=f, fill=(0, 0, 0, 160))
            x += tw + d.textlength(' ', font=f)
        yy += lh
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10)))
    img.alpha_composite(layer)
    d = ImageDraw.Draw(img)
    yy = y
    for ln in lines:
        x = x0
        for seg, hl in ln:
            d.text((x, yy), seg, font=f, fill=(255, 255, 255))
            x += d.textlength(seg + ' ', font=f)
        yy += lh
    img.convert('RGB').save(c['out'], quality=95)
    print('ok', c['out'])

if __name__ == '__main__':
    for c in json.load(open(sys.argv[1])):
        make(c)
