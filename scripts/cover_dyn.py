#!/usr/bin/env python3
"""Dynamic covers: one style (Inter Black uppercase, white + red, black stroke, soft shadow), a different
composition per short so a batch doesn't look copy-pasted. Built on cover_bold.py.
Layouts (c["layout"]):
  rec     — camera viewfinder corners + crossed-out «● REC», text bottom (memory / video topics)
  ghost   — semi-transparent shifted copies of the speaker behind him (other people / influence)
  zoom    — radial zoom blur background, sharp speaker, text top, key word fades out (something vanishes)
  speed   — horizontal motion-blur background + speed streaks, tilted text, key word on a red block (race)
  behind  — giant words BEHIND the head (speaker cut out in front), small line at the bottom
Common keys: src, t, cx, zoom, prefilter (same as the short), retouch (null = none), out, lines, size, y, tilt.
lines: [[["ТЫ В", 0], ["ГОНКЕ", 2]], ...]  0 white, 1 red, 2 white on red block, 3 red fading out.
Usage: cover_dyn.py covers.json"""
import json, os, subprocess, sys, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin
from cover_bold import W, H, BLACK, RED, matte, bottom_fade

def grab(c):
    src, t = c['src'], c['t']
    wh = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', src],
                        capture_output=True, text=True).stdout.strip().split(',')
    sw, sh = int(wh[0]), int(wh[1])
    cw = min(sw, round(sh * 9 / 16 / 2) * 2); bx = int(min(max(c.get('cx', sw / 2) - cw / 2, 0), sw - cw))
    vf = (c['prefilter'] + ',' if c.get('prefilter') else '') + f'crop={cw}:{sh}:{bx}:0'
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', src, '-map', '0:v:0', '-frames:v', '1', '-vf', vf,
                          '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    if c.get('retouch'): im = skin.Retoucher(c['retouch'])(im)
    z = c.get('zoom', 1.0)
    if z > 1:
        b, _ = skin.yunet_detect(im)
        fx, fy = (b[0] + b[2] / 2, b[1] + b[3] / 2) if b is not None else (cw / 2, sh * 0.37)
        w2, h2 = int(cw / z), int(sh / z)
        x0 = int(min(max(fx - w2 / 2, 0), cw - w2)); y0 = int(min(max(fy - h2 * 0.36, 0), sh - h2))
        im = im[y0:y0 + h2, x0:x0 + w2]
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_LANCZOS4)

def to_pil(bgr): return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')

def cutout(bgr, rgb):
    p = rgb.copy(); p.putalpha(Image.fromarray((matte(bgr) * 255).astype(np.uint8))); return p

def face(bgr):
    b, _ = skin.yunet_detect(bgr)
    return (b[0] + b[2] / 2, b[1] + b[3] / 2, b[2]) if b is not None else (W / 2, H * 0.37, W * 0.4)

def text_layer(lines, size=150, maxw=W - 100, align='center', tilt=0.0):
    """renders the text block on its own transparent layer (so it can be tilted), returns (layer, h)"""
    tmp = ImageDraw.Draw(Image.new('RGBA', (8, 8)))
    def lw(ln, f): return sum(tmp.textlength(s, font=f) for s, _ in ln) + tmp.textlength(' ', font=f) * (len(ln) - 1)
    s = size
    while max(lw(ln, ImageFont.truetype(BLACK, s)) for ln in lines) > maxw and s > 60: s -= 4
    f = ImageFont.truetype(BLACK, s); lh = int(s * 1.1); sw = max(6, s // 16); pad = int(s * 0.18)
    LW, LH = W + 400, lh * len(lines) + 200
    lay = Image.new('RGBA', (LW, LH), (0, 0, 0, 0)); sh = Image.new('RGBA', (LW, LH), (0, 0, 0, 0))
    d, sd = ImageDraw.Draw(lay), ImageDraw.Draw(sh)
    y = 100
    for ln in lines:
        x = 200 + ((W - lw(ln, f)) / 2 if align == 'center' else 50)
        for seg, hl in ln:
            tw = d.textlength(seg, font=f); bb = d.textbbox((x, y), seg, font=f)
            if hl == 2:                                     # white word on a red block
                box = (bb[0] - pad, bb[1] - pad * 0.7, bb[2] + pad, bb[3] + pad * 0.7)
                sd.rounded_rectangle((box[0] + 8, box[1] + 12, box[2] + 8, box[3] + 12), 14, fill=(0, 0, 0, 170))
                d.rounded_rectangle(box, 14, fill=RED)
                d.text((x, y), seg, font=f, fill=(255, 255, 255))
            elif hl == 3:                                   # red word dissolving to the right
                w_ = Image.new('RGBA', (LW, LH), (0, 0, 0, 0))
                ImageDraw.Draw(w_).text((x, y), seg, font=f, fill=RED, stroke_width=sw, stroke_fill=(0, 0, 0))
                a = np.asarray(w_).copy(); xs = np.arange(LW, dtype=np.float32)
                g = np.clip(1 - (xs - (x + tw * 0.35)) / (tw * 0.75), 0.12, 1)
                a[..., 3] = (a[..., 3] * g[None, :]).astype(np.uint8)
                rnd = np.random.default_rng(4)                # a few red particles flying off the end
                for _ in range(60):
                    px = x + tw * rnd.uniform(0.7, 1.35); py = y + s * rnd.uniform(0.1, 1.0); r = rnd.uniform(3, 9)
                    ImageDraw.Draw(w_).rectangle((px, py, px + r, py + r), fill=RED + (int(255 * rnd.uniform(0.2, 0.8)),))
                a2 = np.asarray(w_).copy(); a2[..., 3] = np.maximum(a[..., 3], np.where(xs[None, :] > x + tw, a2[..., 3], 0))
                lay.alpha_composite(Image.fromarray(a2))
                sd.text((x + 6, y + 10), seg, font=f, fill=(0, 0, 0, 120), stroke_width=sw, stroke_fill=(0, 0, 0, 120))
            else:
                sd.text((x + 6, y + 10), seg, font=f, fill=(0, 0, 0, 190), stroke_width=sw, stroke_fill=(0, 0, 0, 190))
                d.text((x, y), seg, font=f, fill=RED if hl == 1 else (255, 255, 255), stroke_width=sw, stroke_fill=(0, 0, 0))
            x += tw + d.textlength(' ', font=f)
        y += lh
    out = sh.filter(ImageFilter.GaussianBlur(12)); out.alpha_composite(lay)
    if tilt: out = out.rotate(tilt, resample=Image.BICUBIC, expand=False)
    return out, lh * len(lines)

def put_text(img, c, y_default):
    lay, h = text_layer(c['lines'], c.get('size', 150), tilt=c.get('tilt', 0), align=c.get('align', 'center'))
    yc = c.get('y', y_default)
    img.alpha_composite(lay, (-200, int(yc - lay.height / 2)))

def radial_zoom(rgb, cx, cy, strength=0.12, n=14):
    acc = np.zeros((H, W, 3), np.float32); a = np.asarray(rgb.convert('RGB')).astype(np.float32)
    for k in range(n):
        s = 1 + strength * k / (n - 1)
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        acc += cv2.warpAffine(a, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray((acc / n).astype(np.uint8)).convert('RGBA')

def make(c):
    bgr = grab(c); rgb = to_pil(bgr); fx, fy, fw = face(bgr); lay = c['layout']
    if lay == 'rec':
        img = bottom_fade(rgb, 0.5, 0.75); d = ImageDraw.Draw(img)
        m, L, t = 70, 120, 12                                   # viewfinder corners
        for (x, y, sx, sy) in [(m, m + 60, 1, 1), (W - m, m + 60, -1, 1), (m, H - m - 560, 1, -1), (W - m, H - m - 560, -1, -1)]:
            d.rectangle((min(x, x + sx * L), y - t // 2, max(x, x + sx * L), y + t // 2), fill=(255, 255, 255))
            d.rectangle((x - t // 2, min(y, y + sy * L), x + t // 2, max(y, y + sy * L)), fill=(255, 255, 255))
        f = ImageFont.truetype(BLACK, 92); rx, ry = m + 44, m + 100
        d.ellipse((rx, ry + 14, rx + 70, ry + 84), fill=RED)
        d.text((rx + 94, ry), "REC", font=f, fill=(255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0))
        d.line((rx - 14, ry + 100, rx + 94 + d.textlength('REC', font=f) + 14, ry - 6), fill=RED, width=12)   # crossed out
        put_text(img, c, 1480)
    elif lay == 'ghost':
        person = cutout(bgr, rgb)
        bg = ImageEnhance.Brightness(rgb.filter(ImageFilter.GaussianBlur(18))).enhance(0.6)
        img = bg.copy()
        for dx, dy, sc, al in c.get('ghosts', [(-380, 60, 0.88, 0.38), (380, 60, 0.88, 0.38), (-200, 0, 0.95, 0.5), (200, 0, 0.95, 0.5)]):
            g = person.resize((int(W * sc), int(H * sc)), Image.LANCZOS)
            g = ImageEnhance.Color(g).enhance(0.2); a = np.asarray(g).copy(); a[..., 3] = (a[..., 3] * al).astype(np.uint8)
            img.alpha_composite(Image.fromarray(a), (int((W - g.width) / 2 + dx), int(dy + H * (1 - sc) * 0.3)))
        img.alpha_composite(person)
        img = bottom_fade(img, 0.55, 0.7)
        put_text(img, c, 1500)
    elif lay == 'zoom':
        person = cutout(bgr, rgb)
        img = radial_zoom(rgb, fx, fy, c.get('zoom_strength', 0.14))
        img = ImageEnhance.Brightness(img).enhance(0.8); img.alpha_composite(person)
        top = np.asarray(img).astype(np.float32); yy = np.arange(H, dtype=np.float32)[:, None]
        g = np.clip(1 - yy / (H * 0.32), 0, 1) ** 1.3; top[..., :3] *= (1 - 0.6 * g)[..., None]   # dark fade at top for text
        img = bottom_fade(Image.fromarray(top.astype(np.uint8), 'RGBA'), 0.7, 0.5)
        put_text(img, c, 330)
    elif lay == 'speed':
        person = cutout(bgr, rgb)
        a = np.asarray(rgb.convert('RGB')); k = np.zeros((1, 81), np.float32); k[0] = 1 / 81
        bg = Image.fromarray(cv2.filter2D(a, -1, k)).convert('RGBA'); bg = ImageEnhance.Brightness(bg).enhance(0.75)
        d = ImageDraw.Draw(bg); rnd = random.Random(7)
        for _ in range(38):                                     # speed streaks
            y = rnd.uniform(0, H); x = rnd.uniform(-200, W); L = rnd.uniform(200, 600)
            d.line((x, y, x + L, y), fill=(255, 255, 255, int(rnd.uniform(40, 120))), width=int(rnd.uniform(2, 6)))
        bg = bg.filter(ImageFilter.GaussianBlur(1.2)); bg.alpha_composite(person)
        img = bottom_fade(bg, 0.5, 0.75)
        put_text(img, c, 1450)
    elif lay == 'behind':
        m = matte(bgr); rows = np.where((m > 0.5).sum(1) > 40)[0]
        top = int(rows[0]) if len(rows) else 440
        d = max(0, c.get('head_top', 440) - top)          # move the frame down so the giant words stay readable
        if d:                                              # the strip that appears on top: reflected + blurred sky
            bgr = cv2.copyMakeBorder(bgr, d, 0, 0, 0, cv2.BORDER_REFLECT)[:H]
            bgr[:d + 40] = cv2.GaussianBlur(bgr[:d + 40], (0, 0), 25)
            m = np.concatenate([np.zeros((d, W), np.float32), m])[:H]
            rgb = to_pil(bgr)
        person = rgb.copy(); person.putalpha(Image.fromarray((m * 255).astype(np.uint8)))
        img = bottom_fade(ImageEnhance.Brightness(rgb).enhance(0.85), 0.5, 0.75)
        big, _ = text_layer(c['big'], c.get('big_size', 260), maxw=W - 60, tilt=c.get('big_tilt', 0))
        img.alpha_composite(big, (-200, int(c.get('big_y', 420) - big.height / 2)))
        img.alpha_composite(person)
        img = bottom_fade(img, 0.62, 0.55)
        put_text(img, c, 1560)
    img.convert('RGB').save(c['out'], quality=95); print('ok', c['out'])

if __name__ == '__main__':
    for c in json.load(open(sys.argv[1])): make(c)
