"""Effects layer for make_short.py: transitions, image inserts, pop-up text, progress bar,
watermark, sound effects and background music with ducking.

All events are anchored in OUTPUT time (seconds of the finished short). Config anchors:
  "word": "threads"   -> first word of the short starting with this text (case-insensitive)
  "at": 123.4         -> SOURCE time (seconds in the original video)
  "out": 5.0          -> output time directly
"""
import os, json, subprocess, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
W, H = 1080, 1920

def asset(p, kind):
    """resolve 'whoosh' -> assets/sfx/whoosh.wav, or a path as given"""
    if p is None: return None
    if os.path.exists(p): return p
    for ext in ('', '.wav', '.mp3', '.png', '.jpg', '.webp', '.cube'):
        q = os.path.join(ROOT, 'assets', kind, p + ext)
        if os.path.exists(q): return q
    raise FileNotFoundError(f'asset not found: {p} ({kind})')

def font(size, weight='Black'):
    for p in (f'/usr/share/fonts/opentype/inter/Inter-{weight}.otf', os.path.join(ROOT, 'assets/fonts', f'Inter-{weight}.otf')):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def ease_out_back(p, s=1.7):
    p -= 1; return 1 + (s + 1) * p ** 3 + s * p ** 2

class FX:
    def __init__(self, G, sh, words, src2out, total, parts, fps=30, face_out=(340, 1000, 560), cap_top=1280):
        self.fps, self.total = fps, total
        self.head_top, self.chin, self.brows = face_out
        self.cap_top = cap_top
        self.src2out = src2out
        self.words = words
        fx = dict(G.get('fx', {})); fx.update(sh.get('fx', {}))
        self.cfg = fx
        self.part_gap = fx.get('part_gap', 30)
        self.transitions, self.overlays, self.sfx = [], [], []
        auto = fx.get('auto', True)
        # --- automatic transitions where the short stitches separate parts of the source
        if auto and fx.get('part_transition', 'whip'):
            for (pa, pb), (na, nb) in zip(parts[:-1], parts[1:]):
                if na - pb < fx.get('part_gap', 30): continue   # small gap = just a trimmed repeat, plain cut
                tt = src2out(na)
                self.transitions.append(dict(type=fx.get('part_transition', 'whip'), t=tt, dur=0.32))
                self.sfx.append(dict(file=fx.get('part_sfx', 'whoosh'), t=tt - 0.25, gain=-9))
        # --- explicit transitions
        for tr in sh.get('transitions', []):
            tt = self.when(tr)
            if tt is None: continue
            self.transitions.append(dict(type=tr.get('type', 'flash'), t=tt, dur=tr.get('dur', 0.3)))
            if tr.get('sfx', True):
                d = {'flash': 'swipe', 'whip': 'whoosh', 'glitch': 'glitch', 'shake': 'boom', 'zoom': 'swipe'}
                self.sfx.append(dict(file=tr.get('sfx') if isinstance(tr.get('sfx'), str) else d.get(tr.get('type', 'flash'), 'whoosh'),
                                     t=tt - (0.25 if tr.get('type') == 'whip' else 0.03), gain=tr.get('gain', -9)))
        # --- image inserts
        for ins in sh.get('inserts', []):
            tt = self.when(ins)
            if tt is None: continue
            card = self.make_card(asset(ins['img'], 'images'), ins)
            pos, y = ins.get('pos', 'auto'), ins.get('y')
            if pos == 'auto' and y is None:
                # may cover hair/forehead, never the eyes
                top_space = self.brows - 40 - 50; low_space = self.cap_top - self.chin - 20
                if top_space >= low_space: room, y0 = top_space, 50
                else: room, y0 = low_space, self.chin + 10
                if card.shape[0] > room:
                    k = room / card.shape[0]
                    card = cv2.resize(card, (int(card.shape[1] * k), int(card.shape[0] * k)), interpolation=cv2.INTER_AREA)
                y = y0 + (room - card.shape[0]) // 2 if card.shape[0] < room else y0
            self.overlays.append(dict(img=card, t0=tt, t1=min(total, tt + ins.get('dur', 2.0)), pos=pos,
                                      anim=ins.get('anim', 'pop'), y=y))
            if ins.get('sfx', 'pop'): self.sfx.append(dict(file=ins.get('sfx', 'pop'), t=tt, gain=ins.get('gain', -6)))
        # --- pop-up text (big emphasis like "53%")
        for em in sh.get('emphasis', []):
            tt = self.when(em)
            if tt is None: continue
            txt = self.make_text(em['text'], em)
            y = em.get('y')
            if y is None and em.get('pos', 'auto') == 'auto': y = max(40, int(self.head_top - txt.shape[0] * 0.55))
            self.overlays.append(dict(img=txt, t0=tt, t1=min(total, tt + em.get('dur', 1.4)),
                                      pos=em.get('pos', 'auto'), anim=em.get('anim', 'pop'), y=y))
            if em.get('sfx', 'pop'): self.sfx.append(dict(file=em.get('sfx', 'pop'), t=tt, gain=em.get('gain', -6)))
            if em.get('shake'): self.transitions.append(dict(type='shake', t=tt + 0.1, dur=0.35))
        # --- extra sound effects
        for s in sh.get('sfx', []):
            tt = self.when(s)
            if tt is not None: self.sfx.append(dict(file=s['file'], t=tt, gain=s.get('gain', -8)))
        # --- static layers
        self.progress = fx.get('progress_bar', False)
        self.wm = None
        if fx.get('watermark'):
            self.wm = self.make_watermark(fx['watermark'])
        self.music = sh.get('music', G.get('music'))
        self.lut = sh.get('lut', G.get('lut'))

    # ---------------------------------------------------------------- anchors
    def when(self, d):
        if 'out' in d: return float(d['out'])
        if 'at' in d: return self.src2out(float(d['at']))
        if 'word' in d:
            key = d['word'].lower(); nth = d.get('nth', 1)
            hits = [w for w in self.words if w['t'].lower().startswith(key)]
            if len(hits) >= nth: return self.src2out(hits[nth - 1]['s']) + d.get('offset', 0)
            print(f"  ! word '{d['word']}' not found in this short, skipped")
        return None

    # ---------------------------------------------------------------- builders
    def make_card(self, path, o):
        im = Image.open(path).convert('RGBA')
        tw = int(W * o.get('width', 0.62)); th = int(im.height * tw / im.width)
        maxh = int(H * o.get('max_height', 0.22))
        if th > maxh: tw, th = int(tw * maxh / th), maxh
        im = im.resize((tw, th), Image.LANCZOS)
        if o.get('style', 'card') == 'card':
            r = 28; bd = 8; pad = 40
            mask = Image.new('L', (tw, th), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, tw, th), r, fill=255)
            im.putalpha(Image.fromarray(np.minimum(np.asarray(mask), np.asarray(im.split()[3]))))
            cw, ch = tw + 2 * bd + 2 * pad, th + 2 * bd + 2 * pad
            card = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
            sh = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 10, cw - pad, ch - pad + 10), r + bd, fill=(0, 0, 0, 140))
            card.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
            ImageDraw.Draw(card).rounded_rectangle((pad, pad, cw - pad, ch - pad), r + bd, fill=(255, 255, 255, 255))
            card.alpha_composite(im, (pad + bd, pad + bd))
            im = card
        rot = o.get('rotate', random.Random(path).choice([-3, -2, 2, 3]))
        if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
        return np.asarray(im)

    def make_text(self, text, o):
        f = font(o.get('size', 210), o.get('weight', 'Black'))
        d = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        bb = d.textbbox((0, 0), text, font=f, stroke_width=12)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        im = Image.new('RGBA', (tw + 80, th + 80), (0, 0, 0, 0))
        shd = Image.new('RGBA', im.size, (0, 0, 0, 0))
        ImageDraw.Draw(shd).text((40 - bb[0] + 8, 40 - bb[1] + 12), text, font=f, fill=(0, 0, 0, 170), stroke_width=12, stroke_fill=(0, 0, 0, 170))
        im.alpha_composite(shd.filter(ImageFilter.GaussianBlur(10)))
        col = tuple(o.get('color', (255, 255, 255)))
        ImageDraw.Draw(im).text((40 - bb[0], 40 - bb[1]), text, font=f, fill=col, stroke_width=o.get('stroke', 10), stroke_fill=(0, 0, 0))
        rot = o.get('rotate', -4)
        if rot: im = im.rotate(rot, resample=Image.BICUBIC, expand=True)
        return np.asarray(im)

    def make_watermark(self, wm):
        if isinstance(wm, str): wm = {'text': wm}
        f = font(wm.get('size', 44), 'SemiBold')
        d = ImageDraw.Draw(Image.new('RGBA', (1, 1))); bb = d.textbbox((0, 0), wm['text'], font=f)
        im = Image.new('RGBA', (bb[2] + 20, bb[3] + 20), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), wm['text'], font=f, fill=(255, 255, 255, int(255 * wm.get('opacity', 0.55))),
                                stroke_width=2, stroke_fill=(0, 0, 0, 90))
        a = np.asarray(im); pos = wm.get('pos', 'top-left')
        x = W - a.shape[1] - 50 if 'right' in pos else 50
        y = 60 if 'top' in pos else H - a.shape[0] - 150
        return a, x, y

    # ---------------------------------------------------------------- per frame
    @staticmethod
    def blend(frame, rgba, x, y, alpha=1.0):
        h, w = rgba.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1: return
        sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
        a = sub[..., 3:4].astype(np.float32) / 255 * alpha
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (roi * (1 - a) + sub[..., 2::-1].astype(np.float32) * a).astype(np.uint8)

    def apply(self, frame, tt):
        # transitions (geometry / colour)
        for tr in self.transitions:
            p = (tt - (tr['t'] - tr['dur'] / 2)) / tr['dur']
            if not 0 <= p <= 1: continue
            k = 1 - abs(2 * p - 1)  # 0 -> 1 -> 0
            typ = tr['type']
            if typ == 'flash':
                frame[:] = (frame.astype(np.float32) * (1 - 0.85 * k ** 1.5) + 255 * 0.85 * k ** 1.5).astype(np.uint8)
            elif typ == 'whip':
                ks = int(4 + 90 * k) | 1
                kern = np.zeros((1, ks), np.float32); kern[0, :] = 1 / ks
                z = 1 + 0.08 * k
                M = np.float32([[z, 0, (1 - z) * W / 2 + (p - 0.5) * 160], [0, z, (1 - z) * H / 2]])
                frame[:] = cv2.filter2D(cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT), -1, kern)
            elif typ == 'zoom':
                z = 1 + 0.18 * k
                M = np.float32([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H * 0.4]])
                frame[:] = cv2.GaussianBlur(cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT), (0, 0), 0.1 + 6 * k)
            elif typ == 'glitch':
                r = random.Random(int(tt * 1000)); s = int(24 * k)
                b, g, rr = cv2.split(frame)
                frame[:] = cv2.merge([np.roll(b, s, 1), g, np.roll(rr, -s, 1)])
                for _ in range(int(8 * k)):
                    y = r.randrange(0, H - 40); hh = r.randrange(10, 80)
                    frame[y:y + hh] = np.roll(frame[y:y + hh], r.randint(-60, 60), 1)
            elif typ == 'shake':
                r = random.Random(int(tt * 1000)); a = 22 * (1 - p)
                M = np.float32([[1.04, 0, -W * 0.02 + r.uniform(-a, a)], [0, 1.04, -H * 0.02 + r.uniform(-a, a)]])
                frame[:] = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        # overlays (images / pop text)
        for ov in self.overlays:
            if not ov['t0'] <= tt <= ov['t1']: continue
            a_in = (tt - ov['t0']) / 0.28; a_out = (ov['t1'] - tt) / 0.18
            if ov['anim'] == 'pop':
                sc = ease_out_back(min(1, a_in)) if a_in < 1 else 1.0
                sc = max(0.05, sc) * (1 if a_out >= 1 else 0.9 + 0.1 * a_out)
            else:
                sc = 1.0
            alpha = min(1, a_in * 1.5, a_out)
            img = ov['img']
            if abs(sc - 1) > 0.01:
                img = cv2.resize(img, (max(2, int(img.shape[1] * sc)), max(2, int(img.shape[0] * sc))), interpolation=cv2.INTER_LINEAR)
            y = ov['y'] if ov.get('y') is not None else {'top': 170, 'upper': 330, 'center': (H - img.shape[0]) // 2,
                                                         'bottom': 1120}.get(ov['pos'], 200)
            y += (ov['img'].shape[0] - img.shape[0]) // 2
            self.blend(frame, img, (W - img.shape[1]) // 2, int(y), alpha)
        if self.wm is not None:
            self.blend(frame, *self.wm)
        if self.progress:
            w = int(W * min(1, tt / max(self.total, 0.01)))
            c = self.cfg.get('progress_color', (132, 52, 255))
            frame[H - 12:H, :w] = (c[2], c[1], c[0])
        return frame

    # ---------------------------------------------------------------- audio
    def mix_audio(self, voice_wav, out_wav):
        inputs = ['-i', voice_wav]; fl = []; mix = ['[v]']
        fl.append('[0:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[v][vsc]')
        idx = 1; used_sc = False
        mus = self.music
        if mus:
            if isinstance(mus, str): mus = {'file': mus}
            mf = mus.get('file', 'auto')
            if mf == 'auto':
                md = os.path.join(ROOT, 'assets', 'music')
                files = sorted(f for f in os.listdir(md) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')))
                own = [f for f in files if not f.startswith('placeholder')]
                mf = os.path.join(md, (own or files)[0]) if files else None
            else:
                mf = asset(mf, 'music')
            if mf:
                inputs += ['-stream_loop', '-1', '-i', mf]
                vol = mus.get('volume_db', -20); fo = mus.get('fade_out', 1.5)
                fl.append(f'[{idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{self.total:.3f},asetpts=PTS-STARTPTS,'
                          f'afade=t=in:d=0.8,afade=t=out:st={max(0, self.total - fo):.3f}:d={fo},volume={vol}dB[m0]')
                if mus.get('duck', True):
                    fl.append('[m0][vsc]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=400:makeup=1[m]')
                    used_sc = True
                else:
                    fl.append('[m0]anull[m]')
                mix.append('[m]'); idx += 1
        if not used_sc:
            fl.append('[vsc]anullsink')
        for k, s in enumerate(sorted(self.sfx, key=lambda s: s['t'])):
            path = asset(s['file'], 'sfx'); ms = max(0, int(s['t'] * 1000))
            if s['t'] >= self.total: continue
            inputs += ['-i', path]
            fl.append(f'[{idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms},volume={s.get("gain", -8)}dB[s{k}]')
            mix.append(f'[s{k}]'); idx += 1
        if len(mix) == 1:
            fl.append('[v]anull[out]')
        else:
            fl.append(f'{"".join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,alimiter=limit=0.95[out]')
        fg = out_wav + '.filter'; open(fg, 'w').write(';\n'.join(fl))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', *inputs, '-filter_complex_script', fg, '-map', '[out]', '-c:a', 'pcm_s16le', out_wav], check=True)
        return out_wav

    def summary(self):
        return (f"fx: {len(self.transitions)} transitions, {len(self.overlays)} overlays, {len(self.sfx)} sfx, "
                f"music={'yes' if self.music else 'no'}, progress={'yes' if self.progress else 'no'}")
