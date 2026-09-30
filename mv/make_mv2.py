"""MV generator for 「世代非対称戦」 (lyrics-driven, multi-camera, fast cuts).

usage: python mv/make_mv2.py <audio.mp3> <out.mp4> [--preview t1,t2,...]
"""
import sys, math, subprocess, os
import numpy as np, librosa, imageio_ffmpeg
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
WW, WH = 1920, 1080                      # world canvas
AUDIO, OUT = sys.argv[1], sys.argv[2]
PREVIEW = None
if "--preview" in sys.argv:
    PREVIEW = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]

TITLE = "世代非対称戦"
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
def F(sz): return ImageFont.truetype(FONT, sz)

# ---------------------------------------------------------------- lyrics
BUB, MIL, ICE, Z, SPL = "bub", "mil", "ice", "z", "split"
COL = {BUB: ((255, 196, 60), (255, 60, 160)), MIL: ((40, 230, 215), (120, 140, 255)),
       ICE: ((175, 225, 255), (255, 255, 255)), Z: ((190, 255, 60), (255, 80, 225)),
       SPL: ((255, 255, 255), (255, 90, 90))}
# (start, text, keyword_len_from_end, big word, world)
def L(t, text, k, big, w): return dict(t=t, text=text, k=k, big=big, w=w)
LINES = [
    L(18.2, "バブルの成功　語られレイド", 3, "RAID", BUB), L(20.3, "肩書き残って　中身はフェード", 3, "FADE", BUB),
    L(22.3, "笑い話だけ　美化のコード", 3, "CODE", BUB), L(24.3, "勝った記憶が　歴史をホールド", 4, "HOLD", BUB),
    L(26.5, "ミレニアル設計　キャリアロード", 4, "ROAD", MIL), L(28.5, "意味を探して　迷うノード", 3, "NODE", MIL),
    L(30.5, "選べた前提　与えられモード", 4, "MODE", MIL), L(32.7, "問い直す余裕　最初からロード", 4, "LOAD", MIL),
    L(34.8, "氷河期　無言で耐えたフェーズ", 4, "PHASE", ICE), L(36.7, "席がないまま　続いたレース", 3, "RACE", ICE),
    L(38.7, "能力じゃない　順番のケース", 3, "CASE", ICE), L(40.7, "並んでただけで　消えたネーム", 3, "NAME", ICE),
    L(42.8, "夢を持てって　言われたデイズ", 3, "DAYS", ICE), L(45.0, "余白はゼロ　詰め込むペース", 3, "PACE", ICE),
    L(47.0, "走らされたが　ゴールはベール", 3, "VEIL", ICE), L(49.0, "説明書なしの　長距離ゲーム", 3, "GAME", ICE),
    L(51.2, "Zのスピード　軽やかスウェイ", 4, "SWAY", Z), L(53.2, "合わなきゃ辞める　それもOK", 2, "OK", Z),
    L(55.1, "最適化される　生存プレイ", 3, "PLAY", Z), L(57.4, "逃げも戦略　今の正解", 2, "正解", Z),
    L(59.2, "否定はしない　比較もしない", 4, "NO JUDGE", SPL), L(61.2, "ルールが違う　ただそれだけ", 3, "RULES", SPL),
    L(63.2, "同じ社会で　別の舞台", 2, "STAGE", SPL), L(65.4, "同時進行の　非対称戦", 4, "非対称戦", SPL),
    L(67.7, "バブルはフェード", 3, "FADE", BUB), L(69.7, "ミレニアルロード", 3, "ROAD", MIL),
    L(71.7, "氷河期ホールド", 4, "HOLD", ICE), L(73.7, "Zはスウェイ", 3, "SWAY", Z),
    L(87.6, "終わってないって　言い訳じゃない", 0, "NOT OVER", ICE), L(92.0, "終われなかった　時代のサイン", 3, "SIGN", ICE),
    L(96.5, "回収してない　時間がまだ", 0, "TIME", ICE), L(98.6, "残ってるだけ　それだけだ", 0, "STILL HERE", ICE),
]
LINE_END = {i: (LINES[i + 1]["t"] if i + 1 < len(LINES) else 100.9) for i in range(len(LINES))}
for i in (23,): LINE_END[i] = 67.7
LINE_END[27] = 75.9; LINE_END[29] = 96.4; LINE_END[31] = 101.0
GEN = [("バブル世代", BUB), ("ミレニアル世代", MIL), ("氷河期世代", ICE), ("Z世代", Z)]
DUR = 106.6

# ---------------------------------------------------------------- audio
y, sr = librosa.load(AUDIO, sr=22050, mono=True)
hop = sr // FPS
Sx = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
rms = librosa.feature.rms(S=Sx)[0]; rms /= rms.max() + 1e-9
mel = np.log1p(librosa.feature.melspectrogram(S=Sx ** 2, sr=sr, n_mels=48)); mel /= np.percentile(mel, 99) + 1e-9
bass = np.clip(mel[:6].mean(0), 0, 1)
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop); onset /= onset.max() + 1e-9
_, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
beats = np.array(beats)
def af(a, t): return float(a[min(max(int(t * FPS), 0), len(a) - 1)])
def beat_phase(t):
    i = np.searchsorted(beats, t) - 1
    return (t - beats[i]) if i >= 0 else t
def punch(t): return math.exp(-beat_phase(t) * 7)

# ---------------------------------------------------------------- shots
def build_shots():
    cuts = set([0.0])
    def add_range(a, b, every):
        bs = [x for x in beats if a <= x < b]
        for j, x in enumerate(bs):
            if j % every == 0: cuts.add(round(float(x), 3))
    add_range(0, 18.0, 2); add_range(18.0, 67.7, 2); add_range(67.7, 76, 1)
    add_range(76, 87.4, 1); add_range(87.4, 101, 2); add_range(101, DUR, 4)
    for l in LINES: cuts.add(l["t"])
    for t0 in (67.7, 69.7, 71.7, 73.7, 75.9, 87.4, 101.0):
        cuts.add(t0)
    cs = sorted(cuts)
    # drop cuts that make shots shorter than 0.35s
    out = [cs[0]]
    for c in cs[1:]:
        if c - out[-1] >= 0.35: out.append(c)
    return out
CUTS = build_shots()

CAMS = ["push", "pull", "panL", "panR", "tiltU", "roll", "whip", "dutch", "spin", "push", "panR", "pull"]
def cam_for(k):
    r = np.random.default_rng(k * 7919 + 13)
    return CAMS[k * 5 % len(CAMS)], int(r.integers(0, 10 ** 6))

def line_at(t):
    idx = None
    for i, l in enumerate(LINES):
        if l["t"] <= t < LINE_END[i]: idx = i
    return idx

def world_for(t, shot):
    """which world (and variant) to draw for this shot"""
    i = line_at(t)
    v = shot % 2
    if i is not None:
        w = LINES[i]["w"]
        return w, v
    if t < 18.2:                                   # intro: cycle generations, hero = ice
        order = [BUB, MIL, ICE, Z, ICE, SPL, ICE, MIL]
        return order[shot % len(order)], (shot // 2) % 2
    if t < 87.4:                                   # break
        order = [ICE, BUB, ICE, MIL, ICE, Z, SPL, ICE]
        return order[shot % len(order)], (shot // 2) % 2
    return "dawn", v

# ---------------------------------------------------------------- worlds
_rng = np.random.default_rng(3)
def grad(top, bot):
    a = np.linspace(0, 1, WH)[:, None, None]
    arr = np.array(top)[None, None, :] * (1 - a) + np.array(bot)[None, None, :] * a
    return Image.fromarray(np.repeat(arr, WW, 1).astype(np.uint8))
G_BUB = grad((40, 5, 90), (255, 120, 60)); G_MIL = grad((4, 14, 40), (10, 70, 90))
G_ICE = grad((5, 20, 60), (150, 205, 245)); G_Z = grad((60, 10, 120), (255, 120, 190))
G_DAWN = grad((10, 30, 80), (255, 190, 120))
BLD = [(int(_rng.integers(0, WW - 120)), int(_rng.integers(260, 720)), int(_rng.integers(90, 200))) for _ in range(18)]
NODES = _rng.random((34, 2)) * [WW, WH * 0.85] + [0, 40]
CONF = _rng.random((110, 4))
SNOW = _rng.random((150, 3))

def snow(d, t, n=150, spd=160):
    for x, yy, s in SNOW[:n]:
        px = (x * WW + math.sin(t + yy * 9) * 30 + t * 40) % WW
        py = (yy * WH + t * spd * (0.5 + s)) % WH
        r = 2 + 4 * s
        d.ellipse([px - r, py - r, px + r, py + r], fill=(240, 248, 255))

def world_bub(t, v):
    im = G_BUB.copy(); d = ImageDraw.Draw(im)
    if v == 0:
        d.ellipse([1150, 260, 1670, 780], fill=(255, 215, 90))
        for k in range(8):
            yy = 470 + k * 38; d.rectangle([1150, yy, 1670, yy + 4 + k * 3], fill=(255, 120, 90))
        for x, h, w in BLD:
            top = WH - h
            d.rectangle([x, top, x + w, WH], fill=(30, 10, 60))
            for wy in range(top + 20, WH - 20, 34):
                for wx in range(x + 12, x + w - 12, 28):
                    if int(wx * 7 + wy * 13 + t * 2) % 5 < 2: d.rectangle([wx, wy, wx + 12, wy + 18], fill=(255, 210, 100))
        u = (t % 4) / 4
        pts = [(120 + k * 1500 / 30, 900 - 720 * (k / 30) ** 1.6 - 25 * math.sin(k * 1.7)) for k in range(int(31 * u) + 1)]
        if len(pts) > 1:
            d.line(pts, fill=(255, 225, 90), width=18)
            x, yy = pts[-1]; d.polygon([(x + 40, yy - 40), (x - 10, yy - 30), (x + 30, yy + 10)], fill=(255, 225, 90))
        d.text((200, 120), "¥", font=F(260), fill=(255, 230, 120), stroke_width=6, stroke_fill=(120, 20, 90))
    else:
        for k in range(9):                                                   # rising bars
            h = 160 + 90 * k + 40 * math.sin(t * 5 + k); x = 130 + k * 190
            d.rectangle([x, WH - h, x + 140, WH], fill=(255, 190 - k * 8, 60 + k * 15))
            d.rectangle([x, WH - h, x + 140, WH - h + 14], fill=(255, 250, 200))
        d.text((640, 200), "¥¥¥", font=F(340), fill=(255, 240, 150), stroke_width=8, stroke_fill=(150, 20, 100))
        a = 1 - (t % 2) / 2                                                   # fading 肩書き card
        c = int(255 * a); d.rounded_rectangle([250, 300, 800, 560], 20, fill=(c, c, int(c * .9)))
        d.text((300, 350), "部長", font=F(120), fill=(int(60 * a), 10, int(60 * a)))
        d.text((300, 480), "中身 ??? %", font=F(50), fill=(int(120 * a), 40, int(90 * a)))
    for cx, cy, sp, ph in CONF:                                                # confetti
        px = cx * WW; py = (cy * WH + t * (200 + 300 * sp)) % WH
        d.rectangle([px, py, px + 12, py + 22], fill=(255, 200 + int(55 * ph), 60 + int(150 * sp)))
    for k in range(-10, 11):                                                   # retro grid
        d.line([WW / 2 + k * 90, 900, WW / 2 + k * 480, WH], fill=(255, 60, 170), width=3)
    return im

def world_mil(t, v):
    im = G_MIL.copy(); d = ImageDraw.Draw(im)
    if v == 0:                                                                 # road
        vp = (960, 420)
        d.polygon([vp, (-300, WH), (WW + 300, WH)], fill=(20, 30, 50))
        for k in range(14):
            z = (k + (t * 1.2) % 1) / 14; z2 = z ** 2.2
            y0 = vp[1] + (WH - vp[1]) * z2; y1 = vp[1] + (WH - vp[1]) * min(1, ((k + 0.5 + (t * 1.2) % 1) / 14) ** 2.2)
            w0 = 5 + 40 * z2
            d.polygon([(960 - w0, y0), (960 + w0, y0), (960 + w0 * 1.2, y1), (960 - w0 * 1.2, y1)], fill=(60, 235, 220))
        for s in (-1, 1):                                                       # forks & signs
            d.line([vp, (960 + s * 900, 250)], fill=(120, 140, 255), width=8)
            d.rectangle([960 + s * 500 - 110, 200, 960 + s * 500 + 110, 300], fill=(30, 50, 90))
            d.text((960 + s * 500 - 90, 210), "?" * 3, font=F(70), fill=(120, 240, 255))
    else:                                                                       # node graph + loading
        pts = NODES
        for a in range(len(pts)):
            dist = np.hypot(*(pts - pts[a]).T); for_b = np.argsort(dist)[1:3]
            for b in for_b: d.line([tuple(pts[a]), tuple(pts[b])], fill=(30, 110, 130), width=3)
        hi = int(t * 6) % len(pts)
        for a in range(len(pts)):
            r = 14 + (14 if a == hi else 0); c = (255, 255, 255) if a == hi else (40, 230, 215)
            d.ellipse([pts[a][0] - r, pts[a][1] - r, pts[a][0] + r, pts[a][1] + r], fill=c)
        for k in range(7): d.text((150 + k * 260, 150 + (k * 97 + t * 60) % 600), "?", font=F(120), fill=(120, 140, 255))
        u = (t % 2.5) / 2.5
        d.rectangle([260, 900, 1660, 960], outline=(255, 255, 255), width=6)
        d.rectangle([266, 906, 266 + 1388 * u, 954], fill=(40, 230, 215))
        d.text((260, 810), f"LOADING {int(u * 99)}%", font=F(70), fill=(255, 255, 255))
    return im

def world_ice(t, v):
    im = G_ICE.copy(); d = ImageDraw.Draw(im)
    if v == 0:                                                                  # endless queue
        vp = (1250, 470)
        d.polygon([(0, 700), (WW, 700), (WW, WH), (0, WH)], fill=(200, 230, 250))
        for k in range(26):
            z = ((k - (t * 0.6) % 1) / 26); z = max(z, 0.001); s = z ** 1.8
            x = vp[0] - (vp[0] + 100) * s; yy = vp[1] + (860 - vp[1]) * s
            h = 40 + 620 * s; w = h * .32
            col = (10 + int(70 * (1 - s)), 25 + int(70 * (1 - s)), 60 + int(70 * (1 - s)))
            d.rounded_rectangle([x - w / 2, yy - h * .72, x + w / 2, yy], int(w * .3), fill=col)
            d.ellipse([x - w * .3, yy - h, x + w * .3, yy - h * .72], fill=col)
        # the one empty chair, glowing
        d.rectangle([1500, 560, 1560, 760], fill=(255, 250, 220)); d.rectangle([1500, 700, 1700, 730], fill=(255, 250, 220))
        d.rectangle([1640, 730, 1660, 820], fill=(255, 250, 220))
    else:                                                                        # race lanes, veil goal
        for k in range(7):
            yy = 330 + k * 100; d.line([0, yy, WW, yy], fill=(255, 255, 255), width=4)
            xx = (t * (300 + 45 * k) + k * 300) % (WW + 300) - 150
            d.ellipse([xx - 24, yy - 100, xx + 24, yy - 52], fill=(20, 30, 80))
            d.rounded_rectangle([xx - 22, yy - 52, xx + 22, yy - 6], 12, fill=(20, 30, 80))
        for k in range(12):
            x = 1500 + 30 * k + 20 * math.sin(t * 2 + k)
            d.line([x, 0, x + 40 * math.sin(t + k), WH], fill=(255, 255, 255), width=14 - k // 2)
    # ice cracks + snow
    for k in range(6):
        x0 = 200 + k * 300; d.line([x0, WH, x0 + 60, WH - 120, x0 - 30, WH - 220], fill=(255, 255, 255), width=3)
    snow(d, t)
    return im

def world_z(t, v):
    im = G_Z.copy(); d = ImageDraw.Draw(im)
    cols = [(190, 255, 60), (255, 255, 255), (60, 235, 255), (255, 80, 225)]
    for k in range(5):
        pts = [(x, 540 + (170 - 20 * k) * math.sin(x / (240 + 30 * k) + t * (3 + k) + k)) for x in range(0, WW + 40, 40)]
        d.line(pts, fill=cols[k % 4], width=30)
    if v == 0:
        for k in range(14):
            px = (k * 210 + t * (150 + 40 * k)) % (WW + 200) - 100; py = 200 + 110 * math.sin(t * 4 + k)
            r = 30 + 20 * (k % 3); d.ellipse([px - r, py - r, px + r, py + r], fill=cols[k % 4])
        d.text((200, 800), "OK", font=F(300), fill=(255, 255, 255), stroke_width=10, stroke_fill=(120, 0, 160))
    else:
        on = int(t * 2) % 2 == 0
        d.rounded_rectangle([560, 380, 1360, 700], 160, fill=(190, 255, 60) if on else (110, 60, 160))
        kx = 1200 if on else 720; d.ellipse([kx - 130, 410, kx + 130, 670], fill=(255, 255, 255))
        d.text((760, 750), "ON" if on else "OFF", font=F(150), fill=(255, 255, 255), stroke_width=6, stroke_fill=(90, 0, 140))
        for k in range(6): d.rounded_rectangle([90 + k * 300, 120 + 40 * math.sin(t * 5 + k), 260 + k * 300, 320 + 40 * math.sin(t * 5 + k)], 30, fill=(255, 255, 255))
    return im

def world_dawn(t, v):
    im = G_DAWN.copy(); d = ImageDraw.Draw(im)
    u = min(1, max(0, (t - 87.4) / 14))
    sy = 800 - 420 * u
    for k in range(8, 0, -1): d.ellipse([960 - 130 - k * 40, sy - 130 - k * 40, 960 + 130 + k * 40, sy + 130 + k * 40], fill=(255, 200 + k * 4, 120 + k * 8) if False else (255, 210 - k * 3, 150 - k * 6))
    d.ellipse([960 - 130, sy - 130, 960 + 130, sy + 130], fill=(255, 245, 220))
    d.polygon([(0, WH), (0, 760), (350, 560), (700, 780), (1100, 620), (1500, 800), (WW, 650), (WW, WH)], fill=(20, 40, 90))
    if v == 0:
        for k in range(14):
            x = 120 + k * 60 + 8 * math.sin(t + k); d.rounded_rectangle([x, 880, x + 26, 960], 10, fill=(10, 20, 50)); d.ellipse([x - 2, 850, x + 28, 880], fill=(10, 20, 50))
    else:
        cx, cy, r = 1400, 400, 220
        d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * (0.35 + 0.15 * math.sin(t)), fill=(255, 255, 255), width=24)
        d.text((cx - 150, cy - 40), "TIME", font=F(80), fill=(255, 255, 255))
    snow(d, t, 90, 60)
    return im

WORLD = {BUB: world_bub, MIL: world_mil, ICE: world_ice, Z: world_z, "dawn": world_dawn}
def render_world(w, t, v):
    if w == SPL:
        a = WORLD[[BUB, MIL, ICE][int(t) % 3]](t, v); b = WORLD[ICE if int(t) % 3 else Z](t + 3, 1 - v)
        m = Image.new("L", (WW, WH), 0); ImageDraw.Draw(m).polygon([(0, 0), (1250, 0), (700, WH), (0, WH)], fill=255)
        a.paste(b, (0, 0), Image.eval(m, lambda p: 255 - p))
        ImageDraw.Draw(a).line([(1250, 0), (700, WH)], fill=(255, 255, 255), width=10)
        return a
    return WORLD[w](t, v)

# ---------------------------------------------------------------- camera
def ease(u): return 1 - (1 - u) ** 3
def camera(im, cam, seed, u, t):
    r = np.random.default_rng(seed); sgn = 1 if r.random() < .5 else -1
    z, th, cx, cy = 1.15, 0.0, WW / 2, WH / 2
    e = ease(u)
    if cam == "push":  z = 1.0 + .45 * e
    elif cam == "pull": z = 1.5 - .5 * e
    elif cam == "panL": z = 1.2; cx += (250 - 500 * e) * sgn
    elif cam == "panR": z = 1.25; cx += (-250 + 500 * e) * sgn
    elif cam == "tiltU": z = 1.25; cy += 130 - 260 * e
    elif cam == "roll": z = 1.3; th = math.radians((-9 + 18 * u) * sgn)
    elif cam == "whip": z = 1.2; cx += (700 * (1 - e)) * sgn; th = math.radians(6 * (1 - e) * sgn)
    elif cam == "dutch": z = 1.25 + .2 * u; th = math.radians(11 * sgn)
    elif cam == "spin": z = 1.4 + .2 * u; th = math.radians(28 * (1 - e) * sgn)
    z *= 1 + .05 * punch(t) + .04 * af(bass, t)
    cx += math.sin(t * 27) * 5 * (.3 + af(rms, t)); cy += math.cos(t * 31) * 5 * (.3 + af(rms, t))
    c, s = math.cos(th), math.sin(th)
    hw = (W * c + H * abs(s)) / (2 * z); hh = (W * abs(s) + H * c) / (2 * z)
    if hw > WW / 2 or hh > WH / 2: z *= max(hw / (WW / 2), hh / (WH / 2)); hw = (W * c + H * abs(s)) / (2 * z); hh = (W * abs(s) + H * c) / (2 * z)
    cx = min(max(cx, hw), WW - hw); cy = min(max(cy, hh), WH - hh)
    a, b = c / z, -s / z; d_, e_ = s / z, c / z
    return im.transform((W, H), Image.AFFINE, (a, b, cx - W / 2 * a - H / 2 * b, d_, e_, cy - W / 2 * d_ - H / 2 * e_), Image.BILINEAR)

# ---------------------------------------------------------------- overlay
VIG = None
def vignette():
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W]; d = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        VIG = np.clip(1 - .55 * d ** 2.2, .35, 1)[..., None].astype(np.float32)
    return VIG

def text_c(d, xy, s, font, fill, stroke=6, sc=(10, 10, 30), anchor="mm"):
    d.text(xy, s, font=font, fill=fill, stroke_width=stroke, stroke_fill=sc, anchor=anchor)

def overlay(im, t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    i = line_at(t); p = punch(t)
    if i is not None:
        l = LINES[i]; acc, acc2 = COL[l["w"]]; lt = t - l["t"]
        # giant keyword behind (punches on each beat)
        big = l["big"]; sz = int((250 if len(big) <= 6 else 150) * (1 + .12 * p))
        if l["w"] in (ICE, "dawn"): sz = int(sz * 1.1)
        text_c(d, (W // 2 + int(6 * math.sin(t * 40) * p), H // 2 - 70), big, F(sz), acc + (150,), 10, (0, 0, 0, 120))
        # subtitle line: whole line pops in, rhyme word coloured & bigger
        txt = l["text"]; k = l["k"]; head, tail = (txt[:-k], txt[-k:]) if k else (txt, "")
        pop = 1 + .35 * max(0, 1 - lt / 0.18)
        f1, f2 = F(int(52 * pop)), F(int(66 * pop))
        w1, w2 = d.textlength(head, font=f1), d.textlength(tail, font=f2)
        x0 = (W - w1 - w2) / 2; yb = H - 78
        d.rounded_rectangle([x0 - 30, yb - 56, x0 + w1 + w2 + 30, yb + 44], 18, fill=(0, 0, 0, 120))
        d.text((x0, yb), head, font=f1, fill=(255, 255, 255), stroke_width=5, stroke_fill=(0, 0, 0), anchor="lm")
        d.text((x0 + w1, yb), tail, font=f2, fill=acc2 if l["w"] != ICE else (150, 220, 255), stroke_width=6, stroke_fill=(0, 0, 0), anchor="lm")
        # generation tag
        for name, w in GEN:
            if w == l["w"]: text_c(d, (150, 60), name, F(40), acc, 5, (0, 0, 0)); break
        if l["w"] == SPL: text_c(d, (170, 60), "同時進行", F(40), acc, 5, (0, 0, 0))
    else:
        cx = W // 2
        if t < 18.2:
            if t < 4.5:  # title slam
                a = 255 if t < 3.8 else int(255 * (4.5 - t) / .7); s = 1 + max(0, 1 - t / .3) * 1.5
                text_c(d, (cx, H // 2), TITLE, F(int(150 * s)), (255, 255, 255, a), 12, (20, 60, 140, a))
                text_c(d, (cx, H // 2 + 130), "氷河期世代の物語", F(46), (175, 225, 255, a), 5, (0, 0, 40, a))
            else:
                j = int((t - 4.5) / 2.3) % 4; name, w = GEN[[0, 1, 2, 3][j]]; lt = (t - 4.5) % 2.3
                s = 1 + max(0, 1 - lt / .2) * .6
                text_c(d, (cx, H // 2), name, F(int(130 * s)), COL[w][0] + (255,), 10, (0, 0, 0, 200))
        elif t < 87.4:
            if t < 75.9: pass
            j = int((t - 75.9) / 2.9); big = ["世代", "非対称", "同じ社会・別の舞台", "氷河期世代"][min(j, 3)] if t >= 75.9 else ""
            if t >= 75.9:
                text_c(d, (cx, H // 2), big if j < 3 else TITLE, F(int((150 if j != 2 else 90) * (1 + .1 * p))), (255, 255, 255, 255), 10, (20, 60, 140))
        elif t >= 101.0:
            a = int(255 * min(1, (t - 101) / 1.2))
            text_c(d, (cx, H // 2 - 20), TITLE, F(150), (255, 255, 255, a), 12, (20, 60, 140, a))
            text_c(d, (cx, H // 2 + 120), "produced by okita", F(44), (200, 230, 255, a), 5, (0, 0, 40, a))
    return ov

def frame(i):
    t = i / FPS
    s = max(np.searchsorted(CUTS, t, side="right") - 1, 0)
    s_end = CUTS[s + 1] if s + 1 < len(CUTS) else DUR
    u = (t - CUTS[s]) / max(s_end - CUTS[s], 1e-3)
    w, v = world_for(t, s)
    cam, seed = cam_for(s)
    if s_end - CUTS[s] < 0.9 and cam in ("whip", "pull"): cam = "push"
    im = render_world(w, t, v)
    im = camera(im, cam, seed, u, t)
    arr = np.asarray(im, np.float32)
    # hit flash + RGB split on strong onsets
    on = af(onset, t)
    if on > .55:
        sh = int(10 * on)
        arr = np.stack([np.roll(arr[..., 0], sh, 1), arr[..., 1], np.roll(arr[..., 2], -sh, 1)], -1)
    arr = arr * vignette() + 60 * punch(t) * (1 if (t >= 67.7 and t < 76) or on > .8 else .35)
    # shot-start white flash
    arr += 120 * max(0, 1 - (t - CUTS[s]) / .08)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    im = Image.alpha_composite(im, overlay(im, t)).convert("RGB")
    # chorus tags: full-screen colour slam
    if 67.7 <= t < 75.9:
        k = min(int((t - 67.7) / 2.05), 3); col = COL[[BUB, MIL, ICE, Z][k]][0]
        fade = max(0, 1 - (t - (67.7 + k * 2.05)) / .15)
        if fade: im = Image.blend(im, Image.new("RGB", (W, H), col), .55 * fade)
    return im

def render(i): return frame(i).tobytes()

if __name__ == "__main__":
    if PREVIEW:
        for t in PREVIEW:
            frame(int(t * FPS)).save(f"{OUT}_{t:g}.png")
        sys.exit()
    n = int(DUR * FPS)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", AUDIO, "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, b in enumerate(pool.imap(render, range(n), chunksize=8)):
            ff.stdin.write(b)
            if k % 300 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
