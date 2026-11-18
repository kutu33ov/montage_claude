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
    """cfg: {"skin": 0.5 (0 = off), "zones": ["nose_tip", "forehead", ...]} — what the owner asked for."""
    cfg = cfg or {}
    k = cfg.get('skin', 0.5)
    if k: img = smooth_skin(img, box, k, 1, 0.6)
    for z in cfg.get('zones', ['nose_tip']):
        img = fix_zone(img, box, z)
    return img
