#!/usr/bin/env python3
"""Infographic cards for inserts (the owner likes these). Used by fx.py when an insert has "card": {...}.

Kinds:
  {"kind": "bars", "title": "Веб-трафик, 2025", "items": [["Боты", 53], ["Люди", 47]], "unit": "%",
   "source": "Imperva, 2025"}                       horizontal bars, first bar in accent colour
  {"kind": "stat", "value": "1 000 000", "label": "просмотров за сутки", "source": "..."}   big number
  {"kind": "versus", "left": ["Неделя", "1 видео"], "right": ["5 минут", "10 копий"], "title": "..."}
  {"kind": "columns", "title": "...", "items": [["2023", 30], ["2024", 42], ["2025", 53]], "unit": "%"}
CLI preview:  python3 scripts/card.py '{"kind":"stat","value":"53%","label":"трафика — боты"}' out.png
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ACCENT, DARK, BG, TRACK, MUTED = (132, 52, 255), (34, 34, 44), (250, 250, 252), (232, 232, 238), (130, 130, 148)

def F(weight, size):
    for p in (f'/usr/share/fonts/opentype/inter/Inter-{weight}.otf', os.path.join(ROOT, 'assets/fonts', f'Inter-{weight}.otf')):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def fit(d, text, weight, size, maxw):
    while size > 18 and d.textlength(text, font=F(weight, size)) > maxw: size -= 2
    return F(weight, size)

def make(spec, W=900):
    kind = spec.get('kind', 'bars'); acc = tuple(spec.get('accent', ACCENT))
    im = Image.new('RGB', (W, 1400), BG); d = ImageDraw.Draw(im); x0, y = 50, 40
    if spec.get('title'):
        d.text((x0, y), spec['title'], font=fit(d, spec['title'], 'SemiBold', 42, W - 100), fill=MUTED); y += 80
    if kind == 'bars':
        items = spec['items']; unit = spec.get('unit', '')
        mx = spec.get('max', 100 if unit == '%' else max(v for _, v in items))
        for i, (lab, v) in enumerate(items):
            d.text((x0, y), str(lab), font=fit(d, str(lab), 'Bold', 40, W - 2 * x0), fill=DARK); y += 58
            d.rounded_rectangle((x0, y, W - x0, y + 84), 22, fill=TRACK)
            bw = max(150, int((W - 2 * x0) * v / mx))
            d.rounded_rectangle((x0, y, x0 + bw, y + 84), 22, fill=acc if i == 0 else DARK)
            d.text((x0 + 26, y + 16), f'{v:g}{unit}', font=F('Black', 46), fill=(255, 255, 255)); y += 120
    elif kind == 'stat':
        f = fit(d, spec['value'], 'Black', 190, W - 100)
        d.text((x0, y - 10), spec['value'], font=f, fill=acc); y += int(f.size * 1.1)
        if spec.get('label'):
            d.text((x0, y), spec['label'], font=fit(d, spec['label'], 'Bold', 54, W - 100), fill=DARK); y += 80
    elif kind == 'versus':
        cw = (W - 2 * x0 - 30) // 2
        for k, (side, fill, tc) in enumerate(((spec['left'], TRACK, DARK), (spec['right'], acc, (255, 255, 255)))):
            xx = x0 + k * (cw + 30)
            d.rounded_rectangle((xx, y, xx + cw, y + 260), 26, fill=fill)
            d.text((xx + 28, y + 30), side[0], font=fit(d, side[0], 'Black', 68, cw - 56), fill=tc)
            if len(side) > 1: d.text((xx + 28, y + 140), side[1], font=fit(d, side[1], 'Bold', 44, cw - 56), fill=tc)
        y += 290
    elif kind == 'columns':
        items = spec['items']; mx = spec.get('max', max(v for _, v in items)); unit = spec.get('unit', ''); ch = 380
        n = len(items); cw = (W - 2 * x0) // n
        for i, (lab, v) in enumerate(items):
            h = max(20, int(ch * v / mx)); xx = x0 + i * cw + 18
            d.rounded_rectangle((xx, y + ch - h + 50, xx + cw - 36, y + ch + 50), 16, fill=acc if i == n - 1 else DARK)
            t = f'{v:g}{unit}'; f = fit(d, t, 'Bold', 40, cw - 30)
            d.text((xx + (cw - 36 - d.textlength(t, font=f)) / 2, y + ch - h), t, font=f, fill=DARK)
            f2 = fit(d, str(lab), 'SemiBold', 34, cw - 20)
            d.text((xx + (cw - 36 - d.textlength(str(lab), font=f2)) / 2, y + ch + 64), str(lab), font=f2, fill=MUTED)
        y += ch + 120
    if spec.get('source'):
        d.text((x0, y + 4), spec['source'], font=F('SemiBold', 30), fill=MUTED); y += 50
    return im.crop((0, 0, W, y + 36)).convert('RGBA')

if __name__ == '__main__':
    make(json.loads(sys.argv[1])).save(sys.argv[2] if len(sys.argv) > 2 else 'card.png'); print('ok')
