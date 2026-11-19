#!/usr/bin/env python3
"""Bold 9:16 covers (owner's ideas, Nov 2026): BIG uppercase Inter Black, key words red, black stroke.
Layouts:
  plain    — retouched frame + dark bottom fade + text
  memory   — blurred background, a faded "memory" photo behind the speaker that disintegrates into particles
  inset    — someone else's footage on top (e.g. the gorilla experiment) with a red ellipse, speaker cut out in front
Text lines: [["ВСЕ НА ТЕБЯ", 0]], [["СМОТРЯТ", 1], ["👀", 0]] -> 1 = red. Emoji are drawn with Noto Color Emoji.
Usage: cover_bold.py covers.json"""
import json, os, subprocess, sys, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skin

W, H = 1080, 1920
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
BLACK = '/usr/share/fonts/opentype/inter/Inter-Black.otf'
if not os.path.exists(BLACK): BLACK = os.path.join(ROOT, 'assets/fonts/Inter-Black.otf')
EMOJI = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
RED = (255, 38, 38)

def grab(src, t, cx, zoom=1.15):
    """9:16 crop around the face at time t, retouched; face centre at ~36% height"""
    bx = int(min(max(cx - 304, 0), 1920 - 608))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', src, '-frames:v', '1', '-vf', f'crop=608:1080:{bx}:0',
                          '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    im = skin.Retoucher({'skin': 0.5})(im)
    b, lm = skin.yunet_detect(im)
    fx, fy = (b[0] + b[2] / 2, b[1] + b[3] / 2) if b is not None else (304, 400)
    cw, ch = int(608 / zoom), int(1080 / zoom)
    x0 = int(min(max(fx - cw / 2, 0), 608 - cw)); y0 = int(min(max(fy - ch * 0.36, 0), 1080 - ch))
    return cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LANCZOS4)

def matte(bgr):
    import onnxruntime as ort
    s = ort.InferenceSession(os.path.join(ROOT, 'models/modnet/onnx/model.onnx'), providers=['CPUExecutionProvider'])
    h, w = bgr.shape[:2]; mw = 512; mh = int(round(512 * h / w / 32)) * 32
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (mw, mh)).astype(np.float32) / 255
    m = s.run(None, {s.get_inputs()[0].name: ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]})[0][0, 0]
    m = np.clip(cv2.resize(m, (w, h)), 0, 1)
    hard = (m > 0.5).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(hard)
    if n > 1:
        keep = cv2.dilate((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8), np.ones((15, 15), np.uint8))
        m = m * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3)
    return m

def emoji_img(ch, size):
    f = ImageFont.truetype(EMOJI, 109)
    im = Image.new('RGBA', (150, 150), (0, 0, 0, 0)); ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox()); k = size / im.height
    return im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)

def is_emoji(s): return any(ord(c) > 0x2600 for c in s)

def draw_text(img, lines, y_center, size=150, maxw=W - 90):
    d = ImageDraw.Draw(img)
    def lw(ln, f):
        w = 0
        for seg, _ in ln: w += (emoji_img(seg, f.size).width if is_emoji(seg) else d.textlength(seg, font=f))
        return w + d.textlength(' ', font=f) * (len(ln) - 1)
    sizes = []
    for ln in lines:
        s = size
        while lw(ln, ImageFont.truetype(BLACK, s)) > maxw and s > 60: s -= 4
        sizes.append(s)
    s = min(sizes)                                     # one size for the whole block
    f = ImageFont.truetype(BLACK, s); lh = int(s * 1.08); sw = max(6, s // 16)
    y = int(y_center - lh * len(lines) / 2)
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(shadow)
    for ln in lines:                                   # soft shadow
        x = (W - lw(ln, f)) / 2
        for seg, _ in ln:
            if not is_emoji(seg): sd.text((x + 6, y + 10), seg, font=f, fill=(0, 0, 0, 190), stroke_width=sw, stroke_fill=(0, 0, 0, 190))
            x += (emoji_img(seg, s).width if is_emoji(seg) else d.textlength(seg, font=f)) + d.textlength(' ', font=f)
        y += lh
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(12)))
    d = ImageDraw.Draw(img); y = int(y_center - lh * len(lines) / 2)
    for ln in lines:
        x = (W - lw(ln, f)) / 2
        for seg, hl in ln:
            if is_emoji(seg):
                e = emoji_img(seg, int(s * 0.95)); img.alpha_composite(e, (int(x), int(y + s * 0.12))); x += e.width
            else:
                d.text((x, y), seg, font=f, fill=RED if hl else (255, 255, 255), stroke_width=sw, stroke_fill=(0, 0, 0))
                x += d.textlength(seg, font=f)
            x += d.textlength(' ', font=f)
        y += lh

def bottom_fade(img, start=0.55, k=0.7):
    a = np.asarray(img).astype(np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]; g = np.clip((yy - H * start) / (H * (1 - start)), 0, 1) ** 1.2
    a[..., :3] *= (1 - k * g)[..., None]
    return Image.fromarray(a.astype(np.uint8), img.mode)

def disintegrate(card, start=0.55, block=10, seed=3):
    """right part of the card breaks into flying squares that fade out"""
    rnd = random.Random(seed); a = np.asarray(card).copy(); h, w = a.shape[:2]
    out = np.zeros((h + 200, w + 260, 4), np.uint8); out[100:100 + h, 0:w] = a
    x0 = int(w * start)
    out[100:100 + h, x0:w, 3] = 0
    for by in range(0, h, block):
        for bx in range(x0, w, block):
            p = (bx - x0) / (w - x0)
            if rnd.random() < p * 0.55: continue                 # more holes further right
            blk = a[by:by + block, bx:bx + block].copy()
            dx = int(p ** 1.6 * rnd.uniform(40, 240)); dy = int(-p ** 1.4 * rnd.uniform(-60, 140))
            blk[..., 3] = (blk[..., 3] * (1 - p ** 1.3 * 0.8)).astype(np.uint8)
            X, Y = bx + dx, 100 + by + dy
            if 0 <= Y < out.shape[0] - block and 0 <= X < out.shape[1] - block:
                out[Y:Y + blk.shape[0], X:X + blk.shape[1]] = np.where(blk[..., 3:4] > out[Y:Y + blk.shape[0], X:X + blk.shape[1], 3:4], blk,
                                                                         out[Y:Y + blk.shape[0], X:X + blk.shape[1]])
    return Image.fromarray(out)

def child_photo(w=380, h=460, seed=5):
    """an old, faded, blurred photo with the silhouette of a small boy (illustration, not a real photo)"""
    rnd = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.array([196, 205, 190], np.float32); ground = np.array([120, 138, 96], np.float32)
    g = np.clip((yy - h * 0.55) / (h * 0.1), 0, 1)[..., None]
    a = sky * (1 - g) + ground * g
    im = Image.fromarray(a.astype(np.uint8)); d = ImageDraw.Draw(im)
    cx, base = w * 0.52, h * 0.86
    skin_c, shirt, pants, hair = (226, 186, 160), (196, 72, 60), (70, 82, 120), (92, 64, 44)
    d.rounded_rectangle((cx - 34, base - 120, cx - 8, base), 10, fill=pants)          # legs
    d.rounded_rectangle((cx + 8, base - 120, cx + 34, base), 10, fill=pants)
    d.rounded_rectangle((cx - 52, base - 235, cx + 52, base - 105), 30, fill=shirt)    # body
    d.rounded_rectangle((cx - 78, base - 225, cx - 46, base - 135), 14, fill=shirt)    # arms
    d.rounded_rectangle((cx + 46, base - 225, cx + 78, base - 135), 14, fill=shirt)
    d.ellipse((cx - 50, base - 335, cx + 50, base - 228), fill=skin_c)                 # big child head
    d.chord((cx - 54, base - 342, cx + 54, base - 262), 180, 360, fill=hair)
    im = im.filter(ImageFilter.GaussianBlur(7))
    a = np.asarray(im).astype(np.float32)
    a = a * 0.8 + np.array([150, 120, 85]) * 0.2                                      # faded, warm
    a += rnd.normal(0, 9, a.shape[:2])[..., None]                                      # grain
    v = 1 - 0.45 * np.clip(((xx - w / 2) / (w * 0.7)) ** 2 + ((yy - h / 2) / (h * 0.7)) ** 2, 0, 1)
    return Image.fromarray(np.clip(a * v[..., None], 0, 255).astype(np.uint8))

def make(c):
    bgr = grab(c['src'], c['t'], c.get('cx', 1240), c.get('zoom', 1.15))
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
    lay = c.get('layout', 'plain')
    if lay == 'plain':
        img = bottom_fade(rgb)
    elif lay == 'memory2':                       # speaker untouched + blurred childhood photo falling apart
        img = bottom_fade(rgb)
        if c.get('memory_img'):                  # owner's old photo: enlarge, soften, age it
            ph = Image.open(c['memory_img']).convert('RGB'); k = c.get('memory_w', 470) / ph.width
            ph = ph.resize((int(ph.width * k), int(ph.height * k)), Image.LANCZOS).filter(ImageFilter.GaussianBlur(c.get('memory_blur', 2.2)))
            ph = Image.blend(ph, Image.new('RGB', ph.size, (150, 122, 88)), 0.18)
        else:
            ph = child_photo()
        card = Image.new('RGBA', (ph.width + 36, ph.height + 100), (244, 240, 228, 255)); card.paste(ph, (18, 18))
        card = disintegrate(card, start=c.get('memory_break', 0.5), block=9).rotate(c.get('memory_rot', 8), resample=Image.BICUBIC, expand=True)
        sh = Image.new('RGBA', img.size, (0, 0, 0, 0)); a = np.asarray(card)[..., 3]
        sh.paste(Image.new('RGBA', card.size, (0, 0, 0, 120)), (c.get('memory_x', 10) + 12, c.get('memory_y', 120) + 16), Image.fromarray(a))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)))
        img.alpha_composite(card, (c.get('memory_x', 10), c.get('memory_y', 120)))
        if c.get('memory_behind', False):                # speaker in front: particles fly BEHIND the head
            m = matte(bgr)
            person = bottom_fade(rgb); person.putalpha(Image.fromarray((m * 255).astype(np.uint8)))
            img.alpha_composite(person)
    else:
        m = matte(bgr)
        person = rgb.copy(); person.putalpha(Image.fromarray((m * 255).astype(np.uint8)))
        bg = rgb.filter(ImageFilter.GaussianBlur(26)); bg = ImageEnhance.Brightness(bg).enhance(0.55)
        img = bg.convert('RGBA')
        if lay == 'memory':
            mem = grab(c['src'], c.get('memory_t', c['t'] + 7), c.get('cx', 1240), 1.35)
            mem = Image.fromarray(cv2.cvtColor(mem, cv2.COLOR_BGR2RGB)).resize((520, 925)).crop((0, 140, 520, 760))
            mem = ImageEnhance.Color(mem).enhance(0.15); mem = ImageEnhance.Contrast(mem).enhance(0.8)
            sep = Image.new('RGB', mem.size, (112, 86, 58)); mem = Image.blend(mem, sep, 0.28).filter(ImageFilter.GaussianBlur(1.2))
            card = Image.new('RGBA', (mem.width + 50, mem.height + 130), (246, 242, 232, 255)); card.paste(mem, (25, 25))
            card = disintegrate(card).rotate(-7, resample=Image.BICUBIC, expand=True)
            img.alpha_composite(card, (c.get('memory_x', 20), c.get('memory_y', 150)))
            k = c.get('person_scale', 0.86); pw, ph = int(W * k), int(H * k)
            al = np.asarray(person.split()[3]).astype(np.float32)
            al[:, :160] *= np.linspace(0, 1, 160)[None, :] ** 1.5      # soften the cut-off shoulder edge
            person.putalpha(Image.fromarray(al.astype(np.uint8)))
            img.alpha_composite(person.resize((pw, ph), Image.LANCZOS), (W - pw + c.get('person_dx', 60), H - ph))
        elif lay == 'inset':
            ins = Image.open(c['inset']).convert('RGB'); k = W / ins.width
            ins = ins.resize((W, int(ins.height * k)), Image.LANCZOS).convert('RGBA')
            d = ImageDraw.Draw(ins); ex, ey, rx, ry = [v * k for v in c['ellipse']]
            for wdt in range(0, 12):
                d.ellipse((ex - rx - wdt, ey - ry - wdt, ex + rx + wdt, ey + ry + wdt), outline=RED)
            img.alpha_composite(ins, (0, 0))
            shift = c.get('person_dy', 330)
            p2 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); p2.alpha_composite(person, (0, 0))
            p2 = p2.transform((W, H), Image.AFFINE, (1, 0, 0, 0, 1, -shift))
            img.alpha_composite(p2)
        img = bottom_fade(img, 0.6, 0.6)
    draw_text(img, c['lines'], c.get('y', 1540), c.get('size', 150))
    img.convert('RGB').save(c['out'], quality=95); print('ok', c['out'])

if __name__ == '__main__':
    for c in json.load(open(sys.argv[1])): make(c)
