#!/usr/bin/env python3
"""Rank frames of a short by face sharpness + frontal pose + eyes looking into the camera.
Usage: pick_frames.py src.mp4 config.json short_id out_sheet.jpg [top_n]"""
import json, subprocess, sys
import numpy as np, cv2

src, cfgp, sid, out = sys.argv[1:5]
TOP = int(sys.argv[5]) if len(sys.argv) > 5 else 8
c = json.load(open(cfgp)); sh = next(s for s in c['shorts'] if s['id'] == sid)
parts = sh.get('parts') or [[sh['start'], sh['end']]]
sil = json.load(open(c['global']['silences']))
faces = np.array(json.load(open(c['global']['faces'])))
f = faces[(faces[:, 0] >= parts[0][0]) & (faces[:, 0] <= parts[-1][1])]
f = f[np.abs(f[:, 1] - np.median(f[:, 1])) < 150]
cx = float(np.median(f[:, 1])); bx = int(min(max(cx - 304, 0), 1920 - 608))

ffc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def pupil_offset(gray_eye):
    """darkest-blob centroid relative to eye-box centre, in [-1,1] units"""
    e = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    h, w = e.shape
    core = e[int(h * 0.25):int(h * 0.8), int(w * 0.1):int(w * 0.9)]
    thr = np.percentile(core, 12)
    m = (core <= thr).astype(np.uint8)
    M = cv2.moments(m)
    if M['m00'] < 3: return None
    px = M['m10'] / M['m00'] / core.shape[1] * 2 - 1
    py = M['m01'] / M['m00'] / core.shape[0] * 2 - 1
    return px, py

def score(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fs = ffc.detectMultiScale(g, 1.08, 6, minSize=(150, 150))
    if not len(fs): return None
    x, y, w, h = max(fs, key=lambda r: r[2] * r[3])
    face = g[y:y + h, x:x + w]
    sharp = cv2.Laplacian(face, cv2.CV_64F).var()
    up = face[int(h * 0.15):int(h * 0.55)]
    eyes = eyec.detectMultiScale(up, 1.05, 6, minSize=(w // 9, w // 9), maxSize=(w // 3, w // 3))
    eyes = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]
    if len(eyes) < 2: return dict(sharp=sharp, eyes=len(eyes), ok=False)
    eyes = sorted(eyes, key=lambda r: r[0])
    l, r = eyes
    # frontal: eyes level and symmetric around the face centre
    sym = abs(((l[0] + l[2] / 2) + (r[0] + r[2] / 2)) / 2 - w / 2) / w
    tilt = abs((l[1] + l[3] / 2) - (r[1] + r[3] / 2)) / w
    offs = [pupil_offset(up[ey:ey + eh, ex:ex + ew]) for ex, ey, ew, eh in eyes]
    if None in offs: return dict(sharp=sharp, eyes=2, ok=False)
    gx = np.mean([o[0] for o in offs]); gy = np.mean([o[1] for o in offs])
    spread = abs(offs[0][0] - offs[1][0])
    return dict(sharp=sharp, eyes=2, ok=True, sym=sym, tilt=tilt, gx=gx, gy=gy, spread=spread)

# candidate times: only inside kept (spoken) parts of the short
ts = []
for a, b in parts:
    t = a + 0.1
    while t < b - 0.1:
        if not any(s0 + 0.05 < t < s1 - 0.05 for s0, s1 in sil): ts.append(round(t, 2))
        t += 0.2
res = []
for t in ts:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.2f}', '-i', src, '-frames:v', '1', '-vf', f'crop=608:1080:{bx}:0',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    if len(raw) != 608 * 1080 * 3: continue
    img = np.frombuffer(raw, np.uint8).reshape(1080, 608, 3)
    s = score(img)
    if not s or not s.get('ok'): continue
    gaze = np.hypot(s['gx'], s['gy'] * 0.7) + s['spread'] * 0.5
    total = np.log(s['sharp'] + 1) * 1.0 - gaze * 4.0 - s['sym'] * 8 - s['tilt'] * 6
    res.append((total, t, s, img))
res.sort(key=lambda r: -r[0])
# keep variety: candidates at least 1.5 s apart
pick = []
for r in res:
    if all(abs(r[1] - p[1]) > 1.5 for p in pick): pick.append(r)
    if len(pick) >= TOP: break
tiles = []
for total, t, s, img in pick:
    tile = cv2.resize(img[100:800, 50:558], (254, 350))
    cv2.putText(tile, f'{t:.1f}', (6, 24), 0, 0.7, (0, 255, 255), 2)
    cv2.putText(tile, f'sh{s["sharp"]:.0f} g{abs(s["gx"]):.2f}', (6, 340), 0, 0.5, (0, 255, 255), 1)
    tiles.append(tile)
cv2.imwrite(out, np.hstack(tiles))
print(sid, 'checked', len(ts), 'frames, ok', len(res), '->', [p[1] for p in pick])
