#!/usr/bin/env python3
"""Prepare a source video for cutting shorts.

Usage:  python3 scripts/prep.py <google-drive-link | local file> <project_dir>

Creates in <project_dir>:
  src.mp4           the source video
  audio.wav         16 kHz mono audio
  transcript.json   word-level transcript (faster-whisper large-v3-turbo, Russian)
  transcript.txt    readable transcript with timestamps
  silences.json     [[start, end], ...] audio pauses (ffmpeg silencedetect -35 dB, 0.25 s)
  faces.json        [[t, cx, cy, w], ...] face centre per second (for the 9:16 crop)
  config.json       skeleton config for make_short.py (fill in "shorts")
"""
import json, os, re, subprocess, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
src, proj = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(f'{proj}/work', exist_ok=True); os.makedirs(f'{proj}/out', exist_ok=True)
video = f'{proj}/src.mp4'

def run(cmd, **kw):
    print('>', ' '.join(cmd)[:160], flush=True)
    return subprocess.run(cmd, check=True, **kw)

# 1. download / copy
if not os.path.exists(video):
    m = re.search(r'/d/([\w-]{20,})', src) or re.search(r'id=([\w-]{20,})', src)
    if m:  # Google Drive file shared "anyone with the link"
        url = f'https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t'
        run(['curl', '-sSL', '-m', '1800', '-o', video, url])
        head = open(video, 'rb').read(512)
        if b'<html' in head.lower():
            sys.exit('Drive returned an HTML page: the file is not shared as "anyone with the link".')
    else:
        run(['cp', src, video])

# 2. audio
run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vn', '-ac', '1', '-ar', '16000', f'{proj}/audio.wav'])

# 3. transcript
if not os.path.exists(f'{proj}/transcript.json'):
    env = dict(os.environ, HF_HUB_DISABLE_XET='1')
    with open(f'{proj}/transcript.txt', 'w') as fo:
        run([sys.executable, f'{HERE}/transcribe.py', f'{proj}/audio.wav', f'{proj}/transcript.json'], stdout=fo, env=env)

# 4. silences
p = subprocess.run(['ffmpeg', '-i', f'{proj}/audio.wav', '-af', 'silencedetect=n=-35dB:d=0.25', '-f', 'null', '-'],
                   capture_output=True, text=True)
sil, st = [], 0.0
for l in p.stderr.splitlines():
    m = re.search(r'silence_start: ([\d.]+)', l)
    if m: st = float(m.group(1))
    m = re.search(r'silence_end: ([\d.]+)', l)
    if m: sil.append([st, float(m.group(1))])
json.dump(sil, open(f'{proj}/silences.json', 'w'))

# 5. faces (1 per second)
cap = cv2.VideoCapture(video)
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = []
for t in np.arange(0, dur, 1.0):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
    if not ok: continue
    sw = 960 / f.shape[1]
    g = cv2.cvtColor(cv2.resize(f, None, fx=sw, fy=sw), cv2.COLOR_BGR2GRAY)
    fs = cc.detectMultiScale(g, 1.1, 6, minSize=(60, 60))
    if len(fs):
        x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
        faces.append((float(t), float(x + w / 2) / sw, float(y + h / 2) / sw, float(w) / sw))
json.dump(faces, open(f'{proj}/faces.json', 'w'))

# 6. config skeleton
cfgp = f'{proj}/config.json'
if not os.path.exists(cfgp):
    style = json.load(open(f'{HERE}/../styles/default.json'))
    g = dict(style['global'])
    g.update(src=video, transcript=f'{proj}/transcript.json', faces=f'{proj}/faces.json',
             silences=f'{proj}/silences.json', workdir=f'{proj}/work', outdir=f'{proj}/out')
    json.dump({'global': g, 'shorts': []}, open(cfgp, 'w'), ensure_ascii=False, indent=1)
print(f'done: {len(sil)} pauses, {len(faces)} face detections, {dur:.0f}s video')
