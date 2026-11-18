#!/usr/bin/env python3
"""Cut one vertical short: remove pauses, 9:16 face crop, punch-in zooms on cuts,
white captions with thin black outline, hook title, loudness normalisation,
plus effects (scripts/fx.py): transitions, image inserts, pop-up text, sound effects,
background music with ducking, progress bar, watermark, LUT.
Usage: make_short.py config.json <short_id> [--dry]"""
import json, sys, subprocess, re
import numpy as np

cfg_all = json.load(open(sys.argv[1]))
G = cfg_all['global']
sh = next(s for s in cfg_all['shorts'] if s['id'] == sys.argv[2])

SRC, OUT_W, OUT_H, FPS = G['src'], 1080, 1920, 30
SRC_W, SRC_H = 1920, 1080
PAD_IN, PAD_OUT = 0.07, 0.10
ZOOM = G.get('zoom', 1.12)

# ---------- words in range ----------
tr = json.load(open(G['transcript']))
words = [w for seg in tr for w in seg['words'] if w['s'] >= sh['start'] - 0.05 and w['e'] <= sh['end'] + 0.3]
fixes = {k.lower(): v for k, v in G.get('fixes', {}).items()}
fixes.update({k.lower(): v for k, v in sh.get('fixes', {}).items()})
for w in words:
    key = re.sub(r'[^\w\-]', '', w['w'].lower())
    w['t'] = fixes.get(key, re.sub(r'[.,!?…«»"]', '', w['w']).lower())
# drop words mapped to empty string (e.g. merged pairs like "тик"+"ток")
words = [w for w in words if w['t']]

# ---------- keep intervals (cut pauses by audio silence) ----------
sil = json.load(open(G['silences']))
parts = sh.get('parts') or [[sh['start'], sh['end']]]
sh['start'], sh['end'] = min(p[0] for p in parts), max(p[1] for p in parts)
keeps = []
for ps, pe in parts:
    cur = ps
    for a, b in sil:
        if b <= ps or a >= pe: continue
        a2, b2 = a + PAD_OUT, b - PAD_IN          # leave a little air around speech
        if b2 - a2 < 0.10: continue
        if a2 > cur: keeps.append([cur, min(a2, pe)])
        cur = max(cur, b2)
    if cur < pe: keeps.append([cur, pe])
# only words that belong to the chosen parts
words = [w for w in words if any(ps <= (w['s'] + w['e']) / 2 <= pe or ps <= w['e'] - 0.1 <= pe and w['s'] >= ps - 0.3 for ps, pe in parts)]
for c in sh.get('cut', []):  # manual cuts [a,b] in source time
    new = []
    for s, e in keeps:
        if e <= c[0] or s >= c[1]: new.append([s, e])
        else:
            if s < c[0]: new.append([s, c[0]])
            if e > c[1]: new.append([c[1], e])
    keeps = new
keeps = [k for k in keeps if k[1] - k[0] > 0.12]

def src2out(t):
    acc = 0.0
    for s, e in keeps:
        if t < s: return acc
        if t <= e: return acc + (t - s)
        acc += e - s
    return acc
total = sum(e - s for s, e in keeps)

# ---------- crop around face ----------
faces = np.array(json.load(open(G['faces'])))
f = faces[(faces[:, 0] >= sh['start']) & (faces[:, 0] <= sh['end'])]
f = f[np.abs(f[:, 1] - np.median(f[:, 1])) < 150]
cx, cy = float(np.median(f[:, 1])) + sh.get('x_shift', 0), float(np.median(f[:, 2]))
cw = round(SRC_H * OUT_W / OUT_H / 2) * 2  # 608
bx = int(min(max(cx - cw / 2, 0), SRC_W - cw))

def zoom_box(z):
    """crop box (x, y, w, h) inside the 9:16 base crop, scaled around the face"""
    w, h = int(cw / z), int(SRC_H / z)
    fx = cx - bx
    x = int(min(max(fx - fx / z, 0), cw - w)); y = int(min(max(cy - cy / z, 0), SRC_H - h))
    return x, y, w, h

# ---------- segment plan ----------
plan, zoomed = [], False
for i, (s, e) in enumerate(keeps):
    # punch-in toggles on jump cuts (skip after very short bits to avoid flicker)
    if i > 0 and keeps[i-1][1] - keeps[i-1][0] > 0.8: zoomed = not zoomed
    plan.append((s, e, ZOOM if zoomed else 1.0))

aparts, labels = [], []
for i, (s, e, z) in enumerate(plan):
    d = e - s; fa = min(0.012, d / 4)
    aparts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS,afade=t=in:d={fa:.3f},afade=t=out:st={d - fa:.3f}:d={fa:.3f}[a{i}]")
    labels.append(f"[a{i}]")
aparts.append(f"{''.join(labels)}concat=n={len(plan)}:v=0:a=1,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
ass_path = f"{G['workdir']}/{sh['id']}.ass"

# ---------- captions (ASS) ----------
def ts(t):
    t = max(t, 0); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

MAXW, MAXC = 3, 16
groups, cur = [], []
for i, w in enumerate(words):
    gap = (w['s'] - words[i - 1]['e']) if i else 0
    if cur and (len(cur) >= MAXW or len(' '.join(x['t'] for x in cur + [w])) > MAXC or gap > 0.45):
        groups.append(cur); cur = []
    cur.append(w)
if cur: groups.append(cur)

ev = []
for gi, g in enumerate(groups):
    a = src2out(g[0]['s']); b = src2out(g[-1]['e']) + 0.08
    if gi + 1 < len(groups):
        nxt = src2out(groups[gi + 1][0]['s'])
        if nxt - b < 0.35: b = nxt
    b = min(b, total)
    if b - a < 0.05: continue
    ev.append(f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{' '.join(x['t'] for x in g)}")
hook = sh.get('hook')
if hook:
    ev.append(f"Dialogue: 1,{ts(0)},{ts(min(3.2, total))},Hook,,0,0,0,,{{\\fad(0,250)}}{hook}")

font = G.get('font', 'Inter')
ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font} ExtraBold,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,80,80,{G.get('cap_margin', 560)},1
Style: Hook,{font} ExtraBold,70,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,8,90,90,130,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(ev) + "\n"
open(ass_path, 'w').write(ass)

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fx import FX, asset
fw = float(np.median(f[:, 3]))
_s = OUT_H / SRC_H   # face geometry in output pixels (at zoom 1.0)
face_out = (int((cy - 0.78 * fw) * _s), int((cy + 0.68 * fw) * _s), int((cy - 0.24 * fw) * _s))  # head top, chin, brows
cap_top = OUT_H - G.get('cap_margin', 560) - 90
fx = FX(G, sh, words, src2out, total, parts, FPS, face_out, cap_top)

def render():
    import cv2
    from skin import detect_face, smooth_skin, fix_nose
    cv2.setNumThreads(2)
    wd = G['workdir']; sid = sh['id']
    afg = f"{wd}/{sid}.afilter"; open(afg, 'w').write(";\n".join(aparts))
    voice = f"{wd}/{sid}.voice.wav"
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', SRC, '-filter_complex_script', afg, '-map', '[aout]', '-c:a', 'pcm_s16le', voice], check=True)
    wav = fx.mix_audio(voice, f"{wd}/{sid}.wav")
    out = f"{G['outdir']}/{sid}.mp4"
    vf = f'ass={ass_path}:fontsdir=/usr/share/fonts'
    if fx.lut: vf = f"lut3d='{asset(fx.lut, 'luts')}'," + vf
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-',
        '-i', wav, '-map', '0:v', '-map', '1:a', '-vf', vf,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', out], stdin=subprocess.PIPE)
    fsz = cw * SRC_H * 3
    acc, box, written = 0.0, None, 0
    for s, e, z in plan:
        acc += e - s
        n = round(acc * FPS) - written
        rd = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{s:.3f}', '-t', f'{e - s:.3f}', '-i', SRC, '-an', '-frames:v', str(n),
            '-vf', f'fps={FPS},crop={cw}:{SRC_H}:{bx}:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        zx, zy, zw, zh = zoom_box(z)
        last = None
        for k in range(n):
            buf = rd.stdout.read(fsz)
            if len(buf) == fsz:
                fr = np.frombuffer(buf, np.uint8).reshape(SRC_H, cw, 3).copy()
                if G.get('skin', True):
                    if k % 8 == 0 or box is None:
                        nb = detect_face(fr)
                        if nb is not None: box = nb
                    fr = smooth_skin(fr, box, **G.get('skin_params', {}))
                    if G.get('nose_fix', True): fr = fix_nose(fr, box)
                last = cv2.resize(fr[zy:zy + zh, zx:zx + zw], (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
            if last is not None:
                enc.stdin.write(fx.apply(last.copy(), written / FPS).tobytes())
                written += 1
        rd.stdout.close(); rd.wait()
    enc.stdin.close(); enc.wait()

if '--dry' not in sys.argv:
    render()
print(f"{sh['id']}: {len(keeps)} cuts, {sh['end'] - sh['start']:.1f}s -> {total:.1f}s, crop cx={cx:.0f}; {fx.summary()}")
