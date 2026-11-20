import cv2, numpy as np
_cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    h, w = img.shape[:2]; s = 480 / w
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    fs = _cc.detectMultiScale(g, 1.1, 5, minSize=(50, 50))
    if not len(fs): return None
    x, y, fw, fh = max(fs, key=lambda r: r[2] * r[3])
    return np.array([x, y, fw, fh]) / s

def smooth_skin(img, box, strength=0.5, passes=1, texture=0.6):
    """Edge-preserving smoothing on skin pixels inside an enlarged face box."""
    if box is None: return img
    H, W = img.shape[:2]
    x, y, w, h = box
    x0, y0 = int(max(x - 0.25 * w, 0)), int(max(y - 0.35 * h, 0))
    x1, y1 = int(min(x + 1.25 * w, W)), int(min(y + 1.45 * h, H))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0: return img
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 135, 80), (255, 180, 130))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 6).astype(np.float32)[..., None] / 255 * strength
    small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if passes >= 2:
        small = cv2.bilateralFilter(small, 0, 45, 4.5)
        small = cv2.bilateralFilter(small, 0, 30, 2.5)
    else:
        small = cv2.bilateralFilter(small, 0, 35, 3)
    sm = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    # bring back a bit of fine luminance texture so skin doesn't look plastic
    tex = roi.astype(np.float32) - cv2.GaussianBlur(roi, (0, 0), 1.2).astype(np.float32)
    sm = sm.astype(np.float32) + texture * tex
    out = roi.astype(np.float32) * (1 - m) + sm * m
    img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return img


# retouch zones, relative to the Haar face box: (centre x, centre y, half-width, half-height)
ZONES = {
    'nose_tip':    (0.53, 0.575, 0.115, 0.085),
    'nose':        (0.52, 0.50, 0.12, 0.17),
    'forehead':    (0.50, 0.13, 0.30, 0.11),
    'cheek_left':  (0.28, 0.60, 0.12, 0.12),   # left as seen by the viewer
    'cheek_right': (0.74, 0.60, 0.12, 0.12),
    'between_brows': (0.52, 0.33, 0.08, 0.06),
    # wide search area for red dots: whole nose incl. where it moves on head turns, above the lips
    'nose_wide':   (0.53, 0.52, 0.23, 0.165),
}

def fix_nose(img, box, red=0.85, smooth=0.9):
    return fix_zone(img, box, 'nose_tip', red, smooth)

def fix_zone(img, box, zone='nose_tip', red=0.85, smooth=0.9):
    """Strong local retouch in one zone: pull excess redness to the surrounding skin tone and smooth hard."""
    if box is None: return img
    H, W = img.shape[:2]
    x, y, w, h = box
    zx, zy, zw, zh = ZONES[zone] if isinstance(zone, str) else zone
    ncx, ncy, ax, ay = x + zx * w, y + zy * h, zw * w, zh * h
    x0, y0 = int(max(ncx - ax * 1.4, 0)), int(max(ncy - ay * 1.4, 0))
    x1, y1 = int(min(ncx + ax * 1.4, W)), int(min(ncy + ay * 1.4, H))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0: return img
    # elliptical region mask, feathered
    ell = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(ell, (int(ncx - x0), int(ncy - y0)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    ell = cv2.GaussianBlur(ell, (0, 0), max(ax, ay) * 0.3)
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = (Y > 70) & (Cr > 135) & (Cr < 185) & (Cb > 80) & (Cb < 130)
    if skin.sum() < 50: return img
    # reference tone: skin right around the tip (sides + above), so the tip matches its neighbours
    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float32)
    dn = ((xx - (ncx - x0)) / ax) ** 2 + ((yy - (ncy - y0)) / ay) ** 2
    ring = skin & (dn > 1.6) & (dn < 4.5) & (yy < (ncy - y0) + 0.3 * ay)
    if ring.sum() < 30: return img
    ref_cr, ref_cb = np.median(Cr[ring]), np.median(Cb[ring])
    excess = Cr - ref_cr
    redness = np.clip(excess / 6.0, 0, 1) * skin * ell
    redness = cv2.GaussianBlur(redness, (0, 0), 2)
    # remove only the excess redness: never go below the surrounding tone
    ycc[..., 1] = Cr - redness * np.clip(excess, 0, None) * red
    ycc[..., 2] = Cb + redness * np.clip(ref_cb - Cb, 0, None) * red * 0.5
    # strong smoothing of luminance inside the nose area (keeps the edges of the nose)
    Ys = cv2.bilateralFilter(Y.astype(np.uint8), 0, 30, 5).astype(np.float32)
    Ys = cv2.bilateralFilter(Ys.astype(np.uint8), 0, 22, 4).astype(np.float32)
    ms = cv2.GaussianBlur((skin * ell).astype(np.float32), (0, 0), 3) * smooth
    ycc[..., 0] = Y * (1 - ms) + Ys * ms
    out = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    img[y0:y1, x0:x1] = out
    return img


def retouch(img, box, cfg=None):
    """cfg: {"skin": 0.5 (0 = off), "zones": ["nose_tip", ...] (redness tone fix),
             "spots": ["nose", ...] (remove small red dots)} — what the owner asked for."""
    cfg = cfg or {}
    k = cfg.get('skin', 0.5)
    if k: img = smooth_skin(img, box, k, 1, 0.6)
    for z in cfg.get('zones', ['nose_tip']):
        img = fix_zone(img, box, z)
    for z in cfg.get('spots', ['nose_wide']):             # small red dots, after the tone fix
        img = fix_spots(img, box, z)
    return img


def fix_spots(img, box, zone='nose', strength=1.0):
    """Remove small red dots (pimples) inside a zone: pixels clearly redder than their immediate
    surroundings get the surroundings' colour and a softened luminance. Keeps the overall tone."""
    if box is None: return img
    H, W = img.shape[:2]
    x, y, w, h = box
    zx, zy, zw, zh = ZONES[zone] if isinstance(zone, str) else zone
    ncx, ncy, ax, ay = x + zx * w, y + zy * h, zw * w * 1.15, zh * h * 1.15
    x0, y0 = int(max(ncx - ax * 1.3, 0)), int(max(ncy - ay * 1.3, 0))
    x1, y1 = int(min(ncx + ax * 1.3, W)), int(min(ncy + ay * 1.3, H))
    roi = img[y0:y1, x0:x1]
    if roi.shape[0] < 20 or roi.shape[1] < 20: return img
    ell = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(ell, (int(ncx - x0), int(ncy - y0)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    ell = cv2.GaussianBlur(ell, (0, 0), max(ax, ay) * 0.15)
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = ((Y > 60) & (Cr > 132) & (Cr < 190)).astype(np.float32)
    k = max(5, int(w * 0.16)) | 1                       # "surroundings" scale ≈ a few spot sizes
    # surroundings = skin only: in side views the wall behind the nose pulled the colour to yellow-grey
    sb = skin > 0
    if sb.sum() < 30: return img
    fill = lambda ch: np.where(sb, ch, np.median(ch[sb])).astype(np.float32)
    base_cr = cv2.medianBlur(np.clip(fill(Cr), 0, 255).astype(np.uint8), k).astype(np.float32)
    base_cb = cv2.medianBlur(np.clip(fill(Cb), 0, 255).astype(np.uint8), k).astype(np.float32)
    base_y = cv2.medianBlur(np.clip(fill(Y), 0, 255).astype(np.uint8), k).astype(np.float32)
    d = Cr - base_cr
    m = np.clip((d - 1.0) / 3.0, 0, 1) * skin * ell
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), max(1.5, w * 0.008)) * strength
    m = np.clip(m * 1.4, 0, 1)
    ycc[..., 1] = Cr * (1 - m) + base_cr * m
    ycc[..., 2] = Cb * (1 - m) + base_cb * m
    ycc[..., 0] = Y * (1 - 0.7 * m) + np.maximum(Y, base_y) * 0.7 * m   # spots are a bit darker: lift them
    img[y0:y1, x0:x1] = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return img


def cheek_ref(img, box):
    """median skin chroma of both cheeks — the tone a healthy nose should have"""
    x, y, w, h = box; vals = []
    for cx in (0.27, 0.77):
        X0, Y0 = int(x + (cx - 0.07) * w), int(y + 0.55 * h); X1, Y1 = int(x + (cx + 0.07) * w), int(y + 0.68 * h)
        X0, Y0 = max(X0, 0), max(Y0, 0)
        p = img[Y0:Y1, X0:X1]
        if p.size == 0: continue
        ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
        ycc = ycc[(ycc[:, 0] > 60) & (ycc[:, 1] > 130) & (ycc[:, 1] < 185)]
        if len(ycc) > 20: vals.append(np.median(ycc[:, 1:], 0))
    return np.mean(vals, 0) if vals else None

def fix_nose_red(img, box, ref, warmth=2.5, strength=1.0, zone='nose_wide', lip_y=None):
    """Pull every pixel of the nose area that is redder than the cheeks (+warmth) down to the cheek tone.
    Works on the whole nose above the lips, so it doesn't depend on exactly where the tip is."""
    if box is None or ref is None: return img
    H, W = img.shape[:2]; x, y, w, h = box
    zx, zy, zw, zh = ZONES[zone] if isinstance(zone, str) else zone
    cx, cy, ax, ay = x + zx * w, y + zy * h, zw * w, zh * h
    X0, Y0, X1, Y1 = int(max(cx - ax * 1.2, 0)), int(max(cy - ay * 1.2, 0)), int(min(cx + ax * 1.2, W)), int(min(cy + ay * 1.2, H))
    roi = img[Y0:Y1, X0:X1]
    if roi.shape[0] < 10 or roi.shape[1] < 10: return img
    ell = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(ell, (int(cx - X0), int(cy - Y0)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
    lip = int(lip_y if lip_y is not None else y + 0.70 * h) - Y0     # never touch the lips
    if lip < ell.shape[0]: ell[max(lip, 0):] = 0
    ell = cv2.GaussianBlur(ell, (0, 0), max(ax, ay) * 0.12)
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    sk = ((Y > 75) & (Cr > 130) & (Cr < 190)).astype(np.float32)   # no shadows / nostrils
    sk = cv2.GaussianBlur(sk, (0, 0), 1.2)
    tcr, tcb = ref[0] + warmth, ref[1]
    ex = Cr - tcr
    m = np.clip(ex / 5.0, 0, 1) * sk * ell
    m = cv2.GaussianBlur(m, (0, 0), max(1.5, w * 0.01)) * strength
    ycc[..., 1] = Cr - m * np.clip(ex, 0, 9)           # cap the shift: bigger pulls turned the nose yellow-green
    # Cb is left alone: pushing it towards the cheeks added blue patches
    img[Y0:Y1, X0:X1] = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return img


class Retoucher:
    """Temporally stable video retouch. Face + 5 landmarks from YuNet EVERY frame (works on 3/4 / side views,
    which the frontal Haar detector missed); box and landmarks smoothed (EMA) and held for up to 1.5 s if
    the face is briefly lost. The nose area is centred on the nose-tip landmark and pulled to the cheek tone
    (+warmth, never pale); small red dots removed; lips excluded using the mouth-corner landmarks."""

    def __init__(self, cfg=None, ema=0.6, ref_ema=0.1, size=None):
        self.cfg = cfg or {}
        self.ema, self.ref_ema = ema, ref_ema
        self.box = None; self.lm = None; self.ref = None; self.lost = 0; self.pgray = None

    def track(self, img):
        b, lm = yunet_detect(img)
        if b is None:
            self.lost += 1
            if self.lost > 45: self.box = self.lm = None
            return self.box
        self.lost = 0
        if self.box is None or np.abs(b[:2] - self.box[:2]).max() > b[2] * 0.6:
            self.box, self.lm = b, lm
        else:
            self.box = self.box * (1 - self.ema) + b * self.ema
            self.lm = self.lm * (1 - self.ema) + lm * self.ema
        return self.box

    def __call__(self, img):
        box = self.track(img)
        if box is None: return img
        x, y, w, h = box; lm = self.lm
        k = self.cfg.get('skin', 0.5)
        # smooth_skin expects a Haar-like box (bigger, includes forehead): expand the YuNet box a bit
        hb = np.array([x - 0.08 * w, y - 0.12 * h, w * 1.16, h * 1.12])
        out = smooth_skin(img, hb, k, 1, 0.6) if k else img.copy()
        if self.cfg.get('tone', 0):                    # subtle whole-face redness evening (owner: "чуть-чуть")
            fr = forehead_ref(out, box, lm)
            if fr is not None:
                self.fref = fr if getattr(self, 'fref', None) is None else self.fref * 0.9 + fr * 0.1
            out = even_tone(out, box, getattr(self, 'fref', None), self.cfg['tone'])
        er, el, tp = lm[0], lm[1], lm[2]; ed = max(np.linalg.norm(el - er), 1)
        profile = abs((tp[0] - er[0]) - (el[0] - tp[0])) / ed > 0.35       # 3/4 or side view: one cheek is hidden
        r = forehead_ref(out, box, lm) if profile else cheek_ref_lm(out, box, lm)
        # never aim below the face's own typical skin tone (side views: forehead/cheek samples hit the wall)
        X0, Y0 = int(max(x, 0)), int(max(y, 0)); fb = out[Y0:int(y + h), X0:int(x + w)]
        if fb.size:
            fy = cv2.cvtColor(fb, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
            fy = fy[(fy[:, 0] > 80) & (fy[:, 1] > 140) & (fy[:, 1] < 185)]
            if len(fy) > 100:
                med = np.median(fy[:, 1:], 0)
                r = med - [3, 0] if r is None else np.array([max(r[0], med[0] - 3), r[1]])
        if r is not None:
            self.ref = r if self.ref is None else self.ref * (1 - self.ref_ema) + r * self.ref_ema
        tip = lm[2]
        zone = ((tip[0] - x) / w, (tip[1] - y) / h - 0.04, 0.24, 0.17)        # nose area around the tip
        lip_y = min(lm[3][1], lm[4][1]) - 0.05 * h
        if self.cfg.get('nose_red', True):
            out = fix_nose_red(out, box, self.ref, self.cfg.get('warmth', 2.5), zone=zone, lip_y=lip_y)
        if self.cfg.get('spots', ['nose_wide']):
            out = fix_spots(out, box, zone, strength=1.2)
        return out

_yunet = None
def yunet_detect(img):
    """YuNet (works on 3/4 and side views). Returns (box[x,y,w,h], landmarks 5x2: r-eye, l-eye, nose tip,
    r-mouth, l-mouth) of the largest face, or (None, None)."""
    global _yunet
    import os
    H, W = img.shape[:2]
    if _yunet is None:
        here = os.path.dirname(os.path.abspath(__file__))
        mp = os.path.join(here, '..', 'models', 'yunet', 'face_detection_yunet_2023mar.onnx')
        _yunet = cv2.FaceDetectorYN.create(mp, '', (W, H), 0.6, 0.3, 50)
    _yunet.setInputSize((W, H))
    _, f = _yunet.detect(img)
    if f is None or not len(f): return None, None
    f = max(f, key=lambda r: r[2] * r[3])
    return np.array(f[:4], np.float64), np.array(f[4:14], np.float64).reshape(5, 2)

def cheek_ref_lm(img, box, lm):
    """skin tone below each eye, above the mouth corner (beard and shadows excluded)"""
    w = box[2]; vals = []
    for eye, mouth in ((lm[0], lm[3]), (lm[1], lm[4])):
        cx = eye[0] + (eye[0] - lm[2][0]) * 0.15; cy = eye[1] * 0.45 + mouth[1] * 0.55
        r = w * 0.07
        p = img[int(max(cy - r, 0)):int(cy + r), int(max(cx - r, 0)):int(cx + r)]
        if p.size == 0: continue
        ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
        ycc = ycc[(ycc[:, 0] > 85) & (ycc[:, 1] > 130) & (ycc[:, 1] < 185)]
        if len(ycc) > 25: vals.append(np.median(ycc[:, 1:], 0))
    return np.median(vals, 0) if vals else None


def forehead_ref(img, box, lm):
    """skin chroma of the forehead (above the eyes) — usually the least red skin on a cold/flushed face"""
    ec = (lm[0] + lm[1]) / 2; ed = np.linalg.norm(lm[1] - lm[0])
    cx, cy, r = ec[0], ec[1] - ed * 0.55, ed * 0.28
    p = img[int(max(cy - r * 0.6, 0)):int(cy + r * 0.6), int(max(cx - r, 0)):int(cx + r)]
    if p.size == 0: return None
    ycc = cv2.cvtColor(p, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
    ycc = ycc[(ycc[:, 0] > 70) & (ycc[:, 1] > 128) & (ycc[:, 1] < 185)]
    return np.median(ycc[:, 1:], 0) if len(ycc) > 30 else None

def even_tone(img, box, ref, strength=0.3, warmth=4.0):
    """take away only `strength` of the redness ABOVE the forehead tone (+warmth) on facial skin;
    everything else (shadows, lips, beard, background) untouched — subtle, no filter look"""
    if ref is None or strength <= 0: return img
    H, W = img.shape[:2]; x, y, w, h = box
    X0, Y0, X1, Y1 = int(max(x - 0.1 * w, 0)), int(max(y - 0.15 * h, 0)), int(min(x + 1.1 * w, W)), int(min(y + 1.05 * h, H))
    roi = img[Y0:Y1, X0:X1]
    if roi.size == 0: return img
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr = ycc[..., 0], ycc[..., 1]
    sk = ((Y > 70) & (Cr > 130) & (Cr < 190)).astype(np.float32)
    ell = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(ell, (int(x + w / 2 - X0), int(y + h * 0.45 - Y0)), (int(w * 0.6), int(h * 0.62)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(sk * ell, (0, 0), max(2, w * 0.03)) * strength
    ex = np.clip(Cr - (ref[0] + warmth), 0, None)
    ycc[..., 1] = Cr - m * ex
    img[Y0:Y1, X0:X1] = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return img
