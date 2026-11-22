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
if sh.get('parts'): sh['start'], sh['end'] = min(p[0] for p in sh['parts']), max(p[1] for p in sh['parts'])

SRC, OUT_W, OUT_H, FPS = G['src'], 1080, 1920, int(G.get('fps', 30))
_wh = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                      '-of', 'csv=p=0', SRC], capture_output=True, text=True).stdout.strip().split(',')
SRC_W, SRC_H = int(_wh[0]), int(_wh[1])          # landscape 1920x1080 or vertical 1080x1920 (from 4K phone footage)
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
# drop words mapped to empty string (e.g. merged pairs like "тик"+"ток") and zero-length hallucinations
for w in words:                 # VAD transcripts give some real words zero length: keep them, give a short span
    if w['e'] - w['s'] < 0.04: w['e'] = w['s'] + 0.15
words = [w for w in words if w['t'] and w['t'] not in ('продолжение', 'следует', 'следует...')]

# ---------- keep intervals (cut pauses by audio silence) ----------
sil = json.load(open(G['silences']))
# outdoors (wind / street) silencedetect finds almost nothing: also treat gaps between words as pauses
_ws = sorted((w for seg in tr for w in seg['words']), key=lambda w: w['s'])
# only when silencedetect found almost nothing (wind/street) — indoors it made extra choppy cuts
if _ws and len(sil) < 4 * (_ws[-1]['e'] - _ws[0]['s']) / 60:
    _gaps = [[a['e'] + 0.12, b['s'] - 0.08] for a, b in zip(_ws, _ws[1:]) if b['s'] - a['e'] > 0.6]
    sil = sorted(sil + [g for g in _gaps if not any(x[0] <= g[0] and g[1] <= x[1] for x in sil)])
parts = sh.get('parts') or [[sh['start'], sh['end']]]
# never clip the last word of a part: whisper end times are early, extend to word end + 0.25 s
_all = sorted((w for seg in tr for w in seg['words']), key=lambda w: w['s'])
_sil = json.load(open(G['silences']))
for p_ in parts:
    if len(p_) > 2: continue        # [a, b, "exact"]: owner-checked end, never extend
    pe0 = p_[1]
    _last = max((i for i, w in enumerate(_all) if w['s'] < pe0 - 0.05), default=None)   # the word the part ends on
    for i_, w_ in ([(_last, _all[_last])] if _last is not None else []):
        if pe0 < w_['e'] + 0.25:
            # end where the voice really goes quiet after this word; else word end + small tail
            q = [a for a, b in _sil if w_['e'] - 0.1 <= a <= w_['e'] + 0.6]
            if q: end = q[0] + 0.15
            else:
                nxt = _all[i_ + 1]['s'] if i_ + 1 < len(_all) else w_['e'] + 0.25
                end = min(w_['e'] + 0.25, nxt - 0.02)      # no pause: don't swallow the next word
            p_[1] = max(pe0, end)
            break
parts = [p_[:2] for p_ in parts]   # drop the "exact" flags after extension
sh['start'], sh['end'] = min(p[0] for p in parts), max(p[1] for p in parts)
keeps = []
for ps, pe in parts:
    cur = ps
    for a, b in sil:
        if b <= ps or a >= pe: continue
        a2, b2 = a + PAD_OUT, b - PAD_IN          # leave a little air around speech
        if b2 - a2 < max(0.10, sh.get('min_pause', G.get('min_pause', 0.25)) - 0.17): continue   # short breaths stay in = fewer jump cuts
        if a2 > cur: keeps.append([cur, min(a2, pe)])
        cur = max(cur, b2)
    if cur < pe: keeps.append([cur, pe])
# only words that belong to the chosen parts
words = [w for w in words if any(w['s'] >= ps - 0.08 and (w['s'] + w['e']) / 2 <= pe + 0.05 for ps, pe in parts)]
for c in sh.get('cut', []):  # manual cuts [a,b] in source time
    new = []
    for s, e in keeps:
        if e <= c[0] or s >= c[1]: new.append([s, e])
        else:
            if s < c[0]: new.append([s, c[0]])
            if e > c[1]: new.append([c[1], e])
    keeps = new
keeps = [k for k in keeps if k[1] - k[0] > 0.12]
# drop pieces where nothing is said (a lone second of silence with a punch-in looks like a glitch)
_ww = [w for seg in tr for w in seg['words']]
keeps = [k for k in keeps if any(w['s'] < k[1] - 0.05 and w['e'] > k[0] + 0.05 for w in _ww)]
# smooth opening (owner's rule): no jump cuts in the first seconds — small pauses there stay in
_SMOOTH = sh.get('smooth_start', G.get('smooth_start', 3.0))
_m, _acc = [], 0.0
for k in keeps:
    if _m and _acc < _SMOOTH and k[0] - _m[-1][1] < 0.9:
        _acc += k[1] - _m[-1][1]; _m[-1] = [_m[-1][0], k[1]]
    else:
        _m.append(list(k)); _acc += k[1] - k[0]
keeps = _m
# captions only for words that are actually heard (manual cuts remove repeats like «и и», «слишком слишком»)
words = [w for w in words if not any(c[0] <= (w['s'] + w['e']) / 2 <= c[1] for c in sh.get('cut', []))]
# glue split-off parts back on: «когда» + «-нибудь» -> «когда-нибудь», «какую» + «-то» -> «какую-то»
glued = []
for w in words:
    if glued and w['t'].startswith('-') and w['s'] - glued[-1]['e'] < 0.4:
        glued[-1] = dict(glued[-1], t=glued[-1]['t'] + w['t'], e=w['e'])
    else:
        glued.append(w)
words = glued

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
cw = min(SRC_W, round(SRC_H * OUT_W / OUT_H / 2) * 2)  # 608 for landscape, full width for vertical
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
    # punch-in only on cuts that remove a real pause (>0.6 s) — tiny cuts stay plain, the picture doesn't jump around
    big_cut = i > 0 and keeps[i][0] - keeps[i-1][1] > 0.6
    if i > 0 and big_cut and keeps[i-1][1] - keeps[i-1][0] > 0.8 and sum(e - s for s, e in keeps[:i]) >= _SMOOTH: zoomed = not zoomed   # no punch-in in the opening
    plan.append((s, e, ZOOM if zoomed else 1.0))
LOOP = sh.get('loop', G.get('loop', False))
if LOOP and plan:   # endless video: the last frame must look like the first one
    plan[-1] = (plan[-1][0], plan[-1][1], plan[0][2])

aparts, labels = [], []
for i, (s, e, z) in enumerate(plan):
    d = e - s; fa = min(0.012, d / 4)
    aparts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS,afade=t=in:d={fa:.3f},afade=t=out:st={d - fa:.3f}:d={fa:.3f}[a{i}]")
    labels.append(f"[a{i}]")
_tot = sum(e - s for s, e, z in plan)
_end = '' if sh.get('loop', G.get('loop', False)) else f",afade=t=out:st={max(0, _tot - 0.2):.3f}:d=0.2"   # soft end, no click
aparts.append(f"{''.join(labels)}concat=n={len(plan)}:v=0:a=1{_end},loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
ass_path = f"{G['workdir']}/{sh['id']}.ass"

# ---------- captions (ASS) ----------
def ts(t):
    t = max(t, 0); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

MAXW, MAXC = 3, 16
# short words that must stay with the NEXT word (never end a caption line)...
PROCLITIC = set('в во к ко на с со у о об обо по за из от до для без под над при про через и а но или да не ни'.split())
# ...and particles that stick to the PREVIOUS word (never start a caption line)
ENCLITIC = {'же', 'ли', 'бы', 'б', 'ж'}
def chars(ws): return len(' '.join(x['t'] for x in ws))
groups, cur = [], []
for i, w in enumerate(words):
    gap = (w['s'] - words[i - 1]['e']) if i else 0
    want_break = bool(cur) and (len(cur) >= MAXW or chars(cur + [w]) > MAXC or gap > 0.45)
    if want_break and gap < 0.6:
        if w['t'] in ENCLITIC and chars(cur + [w]) <= MAXC + 6:
            want_break = False                  # "можно же", "есть ли": particle stays on this line
        elif all(x['t'] in PROCLITIC for x in cur):
            want_break = False                  # line is only "в" / "и на": keep it with its word
    if want_break:
        carry = []
        while len(cur) > 1 and cur[-1]['t'] in PROCLITIC and gap < 0.6:
            carry.insert(0, cur.pop())          # "в" / "на" / "и" go down to the next line with their word
        groups.append(cur)
        cur = carry
    cur.append(w)
if cur: groups.append(cur)

# terms highlighted in light red: "terms": ["эффект невозвратных затрат", "прожектора", ...] (global or per short).
# A multi-word term colours every word of the matched sequence (it may span caption lines).
_c = G.get('term_color', 'FF6B6B'); TERM_ON = '{\\c&H' + _c[4:6] + _c[2:4] + _c[0:2] + '&}'; TERM_OFF = '{\\c&HFFFFFF&}'
_terms = [t.lower().split() for t in (G.get('terms', []) + sh.get('terms', []))]
_flat = [(gi, k) for gi, g in enumerate(groups) for k in range(len(g))]
_lw = [re.sub(r'[^\w-]', '', groups[gi][k]['t'].lower()) for gi, k in _flat]
_hit = set()
for term in _terms:
    stem = [w[:max(4, len(w) - 2)] if len(w) > 4 else w for w in term]     # match Russian word forms
    for i in range(len(_lw) - len(term) + 1):
        if all(_lw[i + j].startswith(stem[j]) for j in range(len(term))):
            _hit.update(_flat[i + j] for j in range(len(term)))
def is_term(gi, k): return (gi, k) in _hit

ev = []
for gi, g in enumerate(groups):
    a = src2out(g[0]['s']); b = src2out(g[-1]['e']) + 0.08
    if gi + 1 < len(groups):
        nxt = src2out(groups[gi + 1][0]['s'])
        if nxt - b < 0.35: b = nxt
    b = min(b, total)
    if b - a < 0.05: continue
    txt = ' '.join(x['t'] for x in g)
    fs = f"{{\\fs{int(78 * 18 / len(txt))}}}" if len(txt) > 18 else ''   # long line: shrink instead of wrapping
    shown = ' '.join((TERM_ON + x['t'] + TERM_OFF) if is_term(gi, k) else x['t'] for k, x in enumerate(g))
    ev.append(f"Dialogue: 0,{ts(a)},{ts(b)},Cap,,0,0,0,,{fs}{shown}")
hook = sh.get('hook')
if LOOP and not sh.get('hook_in_loop', False): hook = None   # a title popping up on every loop gives the seam away
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
fx.loop = sh.get('loop', G.get('loop', False))

def render():
    import cv2
    from skin import Retoucher
    rt = Retoucher(sh.get('retouch', G.get('retouch')), size=float(np.median(f[:, 3])))   # expected face width from faces.json
    cv2.setNumThreads(2)
    wd = G['workdir']; sid = sh['id']
    afg = f"{wd}/{sid}.afilter"; open(afg, 'w').write(";\n".join(aparts))
    voice = f"{wd}/{sid}.voice.wav"
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', SRC, '-filter_complex_script', afg, '-map', '[aout]', '-c:a', 'pcm_s16le', voice], check=True)
    wav = fx.mix_audio(voice, f"{wd}/{sid}.wav")
    out = f"{G['outdir']}/{sid}.mp4"
    vf = f'ass={ass_path}:fontsdir=/usr/share/fonts,scale=out_color_matrix=bt709:out_range=tv'   # RGB->YUV with the HD matrix (was BT.601 -> washed colours)
    if fx.lut: vf = f"lut3d='{asset(fx.lut, 'luts')}'," + vf
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-',
        '-i', wav, '-map', '0:v', '-map', '1:a', '-vf', vf,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', str(G.get('crf', 17)), '-pix_fmt', 'yuv420p', '-profile:v', 'high',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
        '-maxrate', G.get('maxrate', '25M'), '-bufsize', G.get('bufsize', '50M'),   # noisy 60fps footage blew up to 60+ Mbps
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-shortest', out], stdin=subprocess.PIPE)
    fsz = cw * SRC_H * 3
    acc, box, written = 0.0, None, 0
    first, XF = None, sh.get('loop_xfade', G.get('loop_xfade', 0.0))   # owner prefers a hard loop cut (no dissolve)
    n_total = round(total * FPS)
    for s, e, z in plan:
        acc += e - s
        n = round(acc * FPS) - written
        rd = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{s:.3f}', '-t', f'{e - s:.3f}', '-i', SRC, '-an', '-frames:v', str(n),
            '-vf', f'fps={FPS},crop={cw}:{SRC_H}:{bx}:0,scale=in_color_matrix=bt709:in_range=tv', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        zx, zy, zw, zh = zoom_box(z)
        rt.pgray = None   # no optical flow across a jump cut
        last = None
        for k in range(n):
            buf = rd.stdout.read(fsz)
            if len(buf) == fsz:
                fr = np.frombuffer(buf, np.uint8).reshape(SRC_H, cw, 3).copy()
                if G.get('skin', True):
                    fr = rt(fr)          # tracked every frame, corrections fade out slowly (skin.Retoucher)
                dr = G.get('drift', 0.0)        # dynamics without transitions: slow push-in within every segment
                if dr: zx, zy, zw, zh = zoom_box(z * (1 + dr * k / max(n - 1, 1)))
                last = cv2.resize(fr[zy:zy + zh, zx:zx + zw], (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
            if last is not None:
                outf = fx.apply(last.copy(), written / FPS)
                if first is None: first = outf.copy()
                if LOOP and XF > 0 and written >= n_total - XF * FPS:   # dissolve into the first frame -> seamless loop
                    p = (written - (n_total - XF * FPS)) / (XF * FPS)
                    outf = cv2.addWeighted(outf, 1 - p, first, p, 0)
                enc.stdin.write(outf.tobytes())
                written += 1
        rd.stdout.close(); rd.wait()
    enc.stdin.close(); enc.wait()

if '--dry' not in sys.argv:
    render()
print(f"{sh['id']}: {len(keeps)} cuts, {sh['end'] - sh['start']:.1f}s -> {total:.1f}s, crop cx={cx:.0f}; {fx.summary()}")
