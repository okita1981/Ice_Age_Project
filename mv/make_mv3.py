"""MV generator v3 for 「世代非対称戦」: generation characters, kinetic English rhyme words, fighting-game intro.

usage: python mv/make_mv3.py <audio.mp3> <out.mp4> [--preview t1,t2,...]
"""
import sys, math, subprocess
from functools import lru_cache
import numpy as np, librosa, imageio_ffmpeg
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
WW, WH = 1920, 1080
AUDIO, OUT = sys.argv[1], sys.argv[2]
PREVIEW = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")] if "--preview" in sys.argv else None
DUR = 106.6
TITLE = "世代非対称戦"

# ---------------------------------------------------------------- fonts
FP = {"lat": "/usr/share/fonts/opentype/montserrat/Montserrat-Black.otf",
      "jp": "/usr/share/fonts/opentype/mplus/Mplus1-Black.otf",
      "tag": "/usr/share/fonts/opentype/bebas-neue/BebasNeue-Bold.otf",
      "jpb": "/usr/share/fonts/opentype/mplus/Mplus1-Bold.otf"}
@lru_cache(maxsize=None)
def font(k, sz): return ImageFont.truetype(FP[k], int(sz))

BUB, MIL, ICE, Z, SPL = "bub", "mil", "ice", "z", "split"
COL = {BUB: ((255, 196, 60), (255, 60, 160)), MIL: ((40, 230, 215), (140, 150, 255)),
       ICE: ((175, 225, 255), (255, 255, 255)), Z: ((190, 255, 60), (255, 80, 225)), SPL: ((255, 255, 255), (255, 90, 90))}
GEN = [("バブル世代", "BUBBLE ERA", BUB), ("ミレニアル世代", "MILLENNIAL", MIL), ("氷河期世代", "ICE AGE", ICE), ("Z世代", "GEN Z", Z)]

# (start, word, gen, kinetic-style, char action, hand prop, world prop, mood)
def L(t, big, w, st, act, hp="none", wp="none", mood="smile"): return dict(t=t, big=big, w=w, st=st, act=act, hp=hp, wp=wp, mood=mood)
LINES = [
    L(18.2, "RAID", BUB, "slam", "celebrate", "glass", "crowd"), L(20.3, "FADE", BUB, "fade", "idle", "none", "card"),
    L(22.3, "CODE", BUB, "type", "laugh", "glass", "code"), L(24.3, "HOLD", BUB, "lock", "fist", "trophy", "rays"),
    L(26.5, "ROAD", MIL, "slide", "walk", "phone", "none", "meh"), L(28.5, "NODE", MIL, "node", "shrug", "none", "ques", "meh"),
    L(30.5, "MODE", MIL, "spin", "idle", "phone", "menu", "meh"), L(32.7, "LOAD", MIL, "fill", "idle", "none", "loading", "meh"),
    L(34.8, "PHASE", ICE, "freeze", "slump", "case", "frost", "sad"), L(36.7, "RACE", ICE, "slide", "run", "case", "chair", "sad"),
    L(38.7, "CASE", ICE, "lock", "idle", "paper", "number", "sad"), L(40.7, "NAME", ICE, "vanish", "idle", "none", "tag", "sad"),
    L(42.8, "DAYS", ICE, "bounce", "slump", "case", "poster", "sad"), L(45.0, "PACE", ICE, "slide", "run", "case", "calendar", "sad"),
    L(47.0, "VEIL", ICE, "fade", "run", "case", "veil", "sad"), L(49.0, "GAME", ICE, "spin", "idle", "book", "none", "sad"),
    L(51.2, "SWAY", Z, "wave", "dance", "phone", "none"), L(53.2, "OK", Z, "bounce", "celebrate", "none", "toggle"),
    L(55.1, "PLAY", Z, "spin", "dance", "phone", "none"), L(57.4, "正解", Z, "slam", "run", "none", "door"),
    L(59.2, "NO JUDGE", SPL, "glitch", "idle", "none", "versus"), L(61.2, "RULES", SPL, "lock", "idle", "none", "versus"),
    L(63.2, "STAGE", SPL, "slam", "idle", "none", "versus"), L(65.4, "非対称戦", SPL, "slam", "fist", "none", "versus"),
    L(67.7, "FADE", BUB, "fade", "celebrate", "glass", "crowd"), L(69.7, "ROAD", MIL, "slide", "walk", "phone", "none", "meh"),
    L(71.7, "HOLD", ICE, "lock", "fist", "case", "frost", "sad"), L(73.7, "SWAY", Z, "wave", "dance", "phone", "none"),
    L(87.6, "NOT OVER", ICE, "slam", "fist", "case", "dawnfx", "smile"), L(92.0, "SIGN", ICE, "fade", "walk", "case", "dawnfx", "smile"),
    L(96.5, "TIME", ICE, "fill", "idle", "case", "clock", "smile"), L(98.6, "STILL HERE", ICE, "slam", "fist", "case", "dawnfx", "smile"),
]
LINE_END = {i: (LINES[i + 1]["t"] if i + 1 < len(LINES) else 101.0) for i in range(len(LINES))}
LINE_END[23] = 67.7; LINE_END[27] = 75.9; LINE_END[29] = 96.4; LINE_END[31] = 101.0

# ---------------------------------------------------------------- audio
y, sr = librosa.load(AUDIO, sr=22050, mono=True)
hop = sr // FPS
Sx = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
rms = librosa.feature.rms(S=Sx)[0]; rms /= rms.max() + 1e-9
mel = np.log1p(librosa.feature.melspectrogram(S=Sx ** 2, sr=sr, n_mels=48)); mel /= np.percentile(mel, 99) + 1e-9
bass = np.clip(mel[:6].mean(0), 0, 1)
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop); onset /= onset.max() + 1e-9
_, beats = librosa.beat.beat_track(y=y, sr=sr, units="time"); beats = np.array(beats)
def af(a, t): return float(a[min(max(int(t * FPS), 0), len(a) - 1)])
def punch(t):
    i = np.searchsorted(beats, t) - 1
    return math.exp(-((t - beats[i]) if i >= 0 else t) * 7)
def ease(u): u = min(max(u, 0), 1); return 1 - (1 - u) ** 3
def clamp(x, a=0, b=1): return min(max(x, a), b)

def line_at(t):
    idx = None
    for i, l in enumerate(LINES):
        if l["t"] <= t < LINE_END[i]: idx = i
    return idx

# ---------------------------------------------------------------- shots
INTRO_CARDS = [5.2, 7.25, 9.3, 11.35]
def build_cuts():
    cuts = {0.0}
    def rng_(a, b, every):
        bs = [x for x in beats if a <= x < b]
        for j, x in enumerate(bs):
            if j % every == 0: cuts.add(round(float(x), 3))
    rng_(0, 2.1, 1); rng_(2.1, 5.2, 1); rng_(5.2, 13.4, 2); rng_(13.4, 18.2, 1)
    rng_(18.2, 67.7, 2); rng_(67.7, 76, 1); rng_(76, 87.4, 1); rng_(87.4, 101, 2); rng_(101, DUR, 4)
    for l in LINES: cuts.add(l["t"])
    for t0 in INTRO_CARDS + [13.4, 15.4, 18.2, 67.7, 69.7, 71.7, 73.7, 75.9, 87.4, 101.0]: cuts.add(t0)
    cs = sorted(cuts); out = [cs[0]]
    for c in cs[1:]:
        if c - out[-1] >= 0.3: out.append(c)
    return out
CUTS = build_cuts()
CAMS = ["push", "pull", "panL", "panR", "tiltU", "roll", "whip", "dutch", "spin", "push", "panR", "pull"]
def cam_for(k):
    r = np.random.default_rng(k * 7919 + 13); return CAMS[k * 5 % len(CAMS)], int(r.integers(0, 10 ** 6))

# ---------------------------------------------------------------- character rig
SPEC = {
    BUB: dict(skin=(255, 214, 180), top=(255, 255, 255), top2=(255, 60, 160), pants=(110, 60, 190), shoe=(255, 196, 60), hair=(40, 20, 20), scale=1.0),
    MIL: dict(skin=(240, 200, 170), top=(40, 190, 185), top2=(20, 80, 110), pants=(50, 60, 95), shoe=(240, 240, 250), hair=(50, 35, 30), scale=1.0),
    ICE: dict(skin=(232, 198, 170), top=(110, 116, 132), top2=(200, 60, 70), pants=(60, 64, 80), shoe=(35, 30, 35), hair=(35, 30, 40), scale=1.08),
    Z: dict(skin=(250, 216, 190), top=(255, 80, 225), top2=(190, 255, 60), pants=(35, 35, 75), shoe=(245, 245, 255), hair=(190, 255, 60), scale=1.0),
}
def pose(act, t):
    p = dict(bob=0, lean=0, hx=0, tilt=0, lt=(0, 0), rt=(0, 0), la=(0, 0), ra=(0, 0))  # thigh angle, knee bend ; arm angle, elbow bend
    s = math.sin
    if act == "walk":
        ph = t * 6.5; p.update(bob=.012 * abs(s(ph)), lean=.06, lt=(.55 * s(ph), .5 * max(0, s(ph + 1.6))), rt=(-.55 * s(ph), .5 * max(0, -s(ph) + .0)),
                            la=(-.5 * s(ph), .3), ra=(.5 * s(ph), .3))
    elif act == "run":
        ph = t * 11; p.update(bob=.03 * abs(s(ph)), lean=.22, lt=(.9 * s(ph), .9 * max(0, s(ph + 1.5))), rt=(-.9 * s(ph), .9 * max(0, -s(ph + 1.5))),
                           la=(-1.0 * s(ph), 1.3), ra=(1.0 * s(ph), 1.3), tilt=.1)
    elif act == "slump":
        p.update(bob=.006 * s(t * 2), lean=.16, tilt=.3, lt=(.15, 0), rt=(-.15, 0), la=(.1 + .05 * s(t * 2), .2), ra=(-.05, .1), hx=.0)
    elif act == "celebrate":
        p.update(bob=.03 * abs(s(t * 8)), lt=(.15 * s(t * 8), .3), rt=(-.15 * s(t * 8), .3), la=(2.7 + .3 * s(t * 9), .3), ra=(2.9 - .3 * s(t * 9), .3), tilt=-.1)
    elif act == "laugh":
        p.update(bob=.025 * abs(s(t * 10)), lean=-.08 + .05 * s(t * 10), lt=(.1, .2), rt=(-.1, .2), la=(.6, 1.6), ra=(1.8 + .3 * s(t * 10), .8), tilt=-.15)
    elif act == "shrug":
        k = .5 + .5 * s(t * 4); p.update(bob=.006 * s(t * 4), la=(.9 + .3 * k, -1.2), ra=(-.9 - .3 * k, 1.2), lt=(.15, 0), rt=(-.15, 0), tilt=.2 * s(t * 2))
    elif act == "dance":
        ph = t * 7; p.update(bob=.03 * abs(s(ph)), hx=.035 * s(ph), lean=.08 * s(ph), lt=(.35 * s(ph), .4), rt=(-.3 * s(ph), .4 * (1 + s(ph)) / 2),
                          la=(2.2 + .6 * s(ph * 2), .6), ra=(1.4 - .7 * s(ph), 1.0), tilt=.2 * s(ph))
    elif act == "fist":
        p.update(bob=.006 * s(t * 3), lean=-.05, lt=(.2, 0), rt=(-.2, 0), la=(-.15, .2), ra=(2.95, .1), tilt=-.15)
    else:  # idle
        p.update(bob=.006 * s(t * 3), lt=(.16, 0), rt=(-.16, 0), la=(.2 + .04 * s(t * 3), .25), ra=(-.2, .25))
    return p

@lru_cache(maxsize=None)
def _dummy(): return 0

def person(gen, act, t, h, hp="none", mood="smile", alpha=1.0, flip=False):
    """returns RGBA layer with feet anchored at (w/2, h*1.24*S...). Returned (layer, anchor_x, anchor_y) in output px."""
    sp = SPEC[gen]; S = 2; hh = h * S * sp["scale"]
    lw, lh = int(1.0 * hh), int(1.32 * hh)
    lay = Image.new("RGBA", (lw, lh), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    ax, ay = lw / 2, lh - .06 * hh
    def P(x, y): return (ax + x * hh, ay + y * hh)
    def circ(c, r, fill, outline=None, w=0):
        d.ellipse([c[0] - r * hh, c[1] - r * hh, c[0] + r * hh, c[1] + r * hh], fill=fill, outline=outline, width=w)
    def limb(p0, a1, a2, l1, l2, col, wd):
        k = P(p0[0] + l1 * math.sin(a1), p0[1] + l1 * math.cos(a1))
        e = P(p0[0] + l1 * math.sin(a1) + l2 * math.sin(a2), p0[1] + l1 * math.cos(a1) + l2 * math.cos(a2))
        s = P(*p0)
        for cc, ww_ in (((20, 20, 35), wd * hh + 6), (col, wd * hh)):
            d.line([s, k], fill=cc, width=int(ww_)); d.line([k, e], fill=cc, width=int(ww_))
            for q in (s, k, e): d.ellipse([q[0] - ww_ / 2, q[1] - ww_ / 2, q[0] + ww_ / 2, q[1] + ww_ / 2], fill=cc)
        return e
    pz = pose(act, t); ol = (20, 20, 35)
    hip = (pz["hx"], -.47 - pz["bob"]); lean = pz["lean"]
    sh = (hip[0] + math.sin(lean) * .26, hip[1] - math.cos(lean) * .26)
    hd = (sh[0] + math.sin(lean + pz["tilt"]) * .11, sh[1] - math.cos(lean + pz["tilt"]) * .11)
    def leg(side, ang):
        a1, kb = ang; a1 = a1 * side if False else a1
        e = limb((hip[0] + side * .025, hip[1]), a1, a1 - kb, .235, .225, sp["pants"], .085)
        d.ellipse([e[0] - .03 * hh, e[1] - .035 * hh, e[0] + .10 * hh, e[1] + .03 * hh], fill=sp["shoe"], outline=ol, width=2)
    def arm(side, ang, front):
        a1, eb = ang; col = sp["top"] if gen != BUB else sp["top"]
        e = limb((sh[0] + side * .02, sh[1] + .015), a1, a1 + eb, .165, .155, col, .07)
        circ(e, .033, sp["skin"], ol, 2)
        return e
    leg(-1, pz["lt"]); a_back = arm(-1, pz["la"], False)
    # torso
    ws, wh = (.16 if gen == BUB else .115), .095
    tp = [P(sh[0] - ws, sh[1]), P(sh[0] + ws, sh[1]), P(hip[0] + wh, hip[1] + .01), P(hip[0] - wh, hip[1] + .01)]
    if gen == MIL:  # backpack
        d.rounded_rectangle([*P(sh[0] - ws - .07, sh[1] + .01), *P(sh[0] - ws + .03, hip[1] - .02)], int(.03 * hh), fill=(70, 70, 150), outline=ol, width=2)
    d.polygon(tp, fill=sp["top"], outline=ol)
    if gen == BUB:  # shoulder pads, gold chain, tie
        d.polygon([P(sh[0] - ws - .05, sh[1] - .01), P(sh[0] - .04, sh[1] - .035), P(sh[0] - .04, sh[1] + .04), P(sh[0] - ws, sh[1] + .06)], fill=sp["top2"], outline=ol)
        d.polygon([P(sh[0] + ws + .05, sh[1] - .01), P(sh[0] + .04, sh[1] - .035), P(sh[0] + .04, sh[1] + .04), P(sh[0] + ws, sh[1] + .06)], fill=sp["top2"], outline=ol)
        d.polygon([P(sh[0], sh[1] + .01), P(sh[0] + .03, sh[1] + .03), P(sh[0], sh[1] + .17), P(sh[0] - .03, sh[1] + .03)], fill=(255, 196, 60))
    elif gen == MIL:  # hoodie pocket + lanyard badge
        d.rounded_rectangle([*P(hip[0] - .07, hip[1] - .1), *P(hip[0] + .07, hip[1] - .03)], 8, fill=sp["top2"])
        d.line([P(sh[0] - .04, sh[1]), P(sh[0], sh[1] + .13), P(sh[0] + .04, sh[1])], fill=(255, 255, 255), width=3)
        d.rectangle([*P(sh[0] - .03, sh[1] + .13), *P(sh[0] + .03, sh[1] + .19)], fill=(255, 255, 255), outline=ol)
    elif gen == ICE:  # loosened tie, scarf, snow on shoulders
        d.polygon([P(sh[0], sh[1] + .01), P(sh[0] + .025, sh[1] + .04), P(sh[0] + .015, sh[1] + .2), P(sh[0] - .015, sh[1] + .2), P(sh[0] - .025, sh[1] + .04)], fill=sp["top2"])
        d.rounded_rectangle([*P(sh[0] - .1, sh[1] - .025), *P(sh[0] + .1, sh[1] + .04)], int(.02 * hh), fill=(170, 220, 255), outline=ol, width=2)
        for q in (-.09, -.03, .05, .1): circ(P(sh[0] + q, sh[1] - .035), .015, (245, 250, 255))
    elif gen == Z:  # oversized hoodie strings
        d.line([P(sh[0] - .02, sh[1] + .03), P(sh[0] - .02, sh[1] + .13)], fill=sp["top2"], width=5)
        d.line([P(sh[0] + .02, sh[1] + .03), P(sh[0] + .02, sh[1] + .12)], fill=sp["top2"], width=5)
        d.rounded_rectangle([*P(hip[0] - .1, hip[1] - .09), *P(hip[0] + .1, hip[1] + .0)], 10, fill=sp["top"], outline=ol)
    leg(1, pz["rt"])
    # head
    c = P(*hd); r = .088
    if gen == BUB: circ((c[0] - .01 * hh, c[1] - .035 * hh), .115, sp["hair"], ol, 2)                 # big 80s hair
    elif gen == Z: circ((c[0] - .01 * hh, c[1] - .02 * hh), .098, sp["hair"], ol, 2)
    elif gen == MIL: circ((c[0], c[1] - .02 * hh), .092, sp["hair"], ol, 2)
    circ(c, r, sp["skin"], ol, 2)
    if gen == ICE: circ((c[0] - .02 * hh, c[1] - .05 * hh), .078, sp["hair"]); d.arc([c[0] - r * hh, c[1] - r * hh, c[0] + r * hh, c[1] + r * hh], 200, 330, fill=sp["hair"], width=int(.05 * hh))
    ex, ey = c[0] + .03 * hh, c[1] - .003 * hh
    if gen == BUB: d.rounded_rectangle([c[0] - .05 * hh, ey - .018 * hh, c[0] + .095 * hh, ey + .02 * hh], 6, fill=(15, 10, 30))
    else:
        for xx in (-.02, .045): circ((ex + xx * hh - .015 * hh, ey), .011, (20, 20, 40))
    if gen == MIL:
        for xx in (-.02, .045): d.ellipse([ex + xx * hh - .015 * hh - .028 * hh, ey - .028 * hh, ex + xx * hh - .015 * hh + .028 * hh, ey + .028 * hh], outline=(20, 20, 40), width=3)
    if gen == ICE:  # tired eyes: dark bags, worried brows
        for xx in (-.02, .045): d.arc([ex + xx * hh - .04 * hh, ey - .01 * hh, ex + xx * hh + .01 * hh, ey + .04 * hh], 20, 160, fill=(120, 90, 110), width=3)
        d.line([ex - .06 * hh, ey - .035 * hh, ex - .01 * hh, ey - .05 * hh], fill=(30, 25, 35), width=4)
    if gen == Z:  # headphones
        d.arc([c[0] - .1 * hh, c[1] - .12 * hh, c[0] + .1 * hh, c[1] + .08 * hh], 180, 360, fill=sp["top2"], width=int(.02 * hh))
        d.rounded_rectangle([c[0] - .11 * hh, c[1] - .03 * hh, c[0] - .07 * hh, c[1] + .05 * hh], 6, fill=sp["top2"], outline=ol)
    mx, my = c[0] + .012 * hh, c[1] + .05 * hh
    if mood == "smile": d.arc([mx - .035 * hh, my - .025 * hh, mx + .035 * hh, my + .025 * hh], 10, 170, fill=(60, 20, 30), width=3)
    elif mood == "sad": d.arc([mx - .03 * hh, my - .0 * hh, mx + .03 * hh, my + .04 * hh], 190, 350, fill=(60, 20, 30), width=3)
    else: d.line([mx - .025 * hh, my + .01 * hh, mx + .025 * hh, my + .01 * hh], fill=(60, 20, 30), width=3)
    e = arm(1, pz["ra"], True)
    # hand prop (front hand)
    hx, hy = e
    if hp == "glass":
        d.polygon([(hx - 12, hy - 50), (hx + 12, hy - 50), (hx, hy - 20)], fill=(255, 235, 150)); d.line([(hx, hy - 20), (hx, hy + 6)], fill=(255, 235, 150), width=4)
    elif hp == "trophy":
        d.polygon([(hx - 34, hy - 100), (hx + 34, hy - 100), (hx + 18, hy - 40), (hx - 18, hy - 40)], fill=(255, 210, 60), outline=ol)
        d.rectangle([hx - 6, hy - 40, hx + 6, hy - 12], fill=(255, 210, 60)); d.rectangle([hx - 24, hy - 12, hx + 24, hy + 4], fill=(230, 170, 40))
    elif hp == "case":
        d.line([(hx, hy), (hx, hy + 22)], fill=ol, width=5)
        d.rounded_rectangle([hx - 42, hy + 22, hx + 42, hy + 88], 8, fill=(120, 80, 50), outline=ol, width=3)
    elif hp == "phone":
        d.rounded_rectangle([hx - 12, hy - 40, hx + 14, hy + 4], 5, fill=(20, 20, 40), outline=(120, 240, 255), width=3)
    elif hp == "book":
        d.rectangle([hx - 34, hy - 44, hx + 34, hy + 10], fill=(245, 235, 210), outline=ol, width=3)
        d.text((hx, hy - 17), "?", font=font("lat", 40), fill=(120, 60, 60), anchor="mm")
    elif hp == "paper":
        d.rectangle([hx - 26, hy - 48, hx + 26, hy + 8], fill=(255, 255, 255), outline=ol, width=3)
        for k in range(4): d.line([(hx - 18, hy - 36 + k * 11), (hx + 18, hy - 36 + k * 11)], fill=(150, 150, 170), width=3)
    lay = lay.resize((lw // S, lh // S), Image.LANCZOS)
    if flip: lay = lay.transpose(Image.FLIP_LEFT_RIGHT)
    if alpha < 1: lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
    return lay, lay.width / 2, lay.height - .06 * hh / S

def place(im, gen, act, t, h, x, feet, flip=False, hp="none", mood="smile", alpha=1.0):
    d = ImageDraw.Draw(im); w = h * .28
    d.ellipse([x - w, feet - 14, x + w, feet + 14], fill=(0, 0, 0))                       # ground shadow
    lay, ax, ay = person(gen, act, t, h, hp, mood, alpha, flip)
    im.paste(lay, (int(x - ax), int(feet - ay)), lay)

# ---------------------------------------------------------------- backgrounds
_r = np.random.default_rng(3)
def grad(top, bot):
    a = np.linspace(0, 1, WH)[:, None, None]
    return Image.fromarray(np.repeat(np.array(top)[None, None] * (1 - a) + np.array(bot)[None, None] * a, WW, 1).astype(np.uint8))
G = {BUB: grad((40, 5, 90), (255, 120, 60)), MIL: grad((4, 14, 40), (10, 70, 90)), ICE: grad((5, 20, 60), (150, 205, 245)),
     Z: grad((60, 10, 120), (255, 120, 190)), "dawn": grad((10, 30, 80), (255, 190, 120)), "card": grad((10, 10, 25), (30, 30, 60))}
BLD = [(int(_r.integers(0, WW - 120)), int(_r.integers(260, 720)), int(_r.integers(90, 200))) for _ in range(18)]
NODES = _r.random((34, 2)) * [WW, WH * 0.85] + [0, 40]
CONF = _r.random((110, 4)); SNOW = _r.random((150, 3))
FEET = 860

def snow(d, t, n=150, spd=160):
    for x, yy, s in SNOW[:n]:
        px = (x * WW + math.sin(t + yy * 9) * 30 + t * 40) % WW; py = (yy * WH + t * spd * (.5 + s)) % WH; r = 2 + 4 * s
        d.ellipse([px - r, py - r, px + r, py + r], fill=(240, 248, 255))

def bg_bub(t, v):
    im = G[BUB].copy(); d = ImageDraw.Draw(im)
    d.ellipse([1150, 200, 1670, 720], fill=(255, 215, 90))
    for k in range(8): d.rectangle([1150, 410 + k * 38, 1670, 414 + k * 41], fill=(255, 120, 90))
    for x, h, w in BLD:
        top = WH - h; d.rectangle([x, top, x + w, WH], fill=(30, 10, 60))
        for wy in range(top + 20, WH - 20, 34):
            for wx in range(x + 12, x + w - 12, 28):
                if int(wx * 7 + wy * 13 + t * 2) % 5 < 2: d.rectangle([wx, wy, wx + 12, wy + 18], fill=(255, 210, 100))
    d.rectangle([0, FEET - 20, WW, WH], fill=(70, 20, 110))
    for k in range(-10, 11): d.line([WW / 2 + k * 90, FEET - 10, WW / 2 + k * 480, WH], fill=(255, 60, 170), width=3)
    return im, d

def bg_mil(t, v):
    im = G[MIL].copy(); d = ImageDraw.Draw(im)
    if v == 0:
        vp = (960, 380); d.polygon([vp, (-300, WH), (WW + 300, WH)], fill=(20, 30, 50))
        for k in range(14):
            z = (k + (t * 1.2) % 1) / 14; y0 = vp[1] + (WH - vp[1]) * z ** 2.2; y1 = vp[1] + (WH - vp[1]) * min(1, ((k + .5 + (t * 1.2) % 1) / 14) ** 2.2); w0 = 5 + 40 * z ** 2.2
            d.polygon([(960 - w0, y0), (960 + w0, y0), (960 + w0 * 1.2, y1), (960 - w0 * 1.2, y1)], fill=(60, 235, 220))
    else:
        for a in range(len(NODES)):
            dist = np.hypot(*(NODES - NODES[a]).T)
            for b in np.argsort(dist)[1:3]: d.line([tuple(NODES[a]), tuple(NODES[b])], fill=(30, 110, 130), width=3)
        hi = int(t * 6) % len(NODES)
        for a in range(len(NODES)):
            r = 14 + (14 if a == hi else 0); d.ellipse([NODES[a][0] - r, NODES[a][1] - r, NODES[a][0] + r, NODES[a][1] + r], fill=(255, 255, 255) if a == hi else (40, 230, 215))
        d.rectangle([0, FEET - 10, WW, WH], fill=(15, 40, 60))
    return im, d

def bg_ice(t, v):
    im = G[ICE].copy(); d = ImageDraw.Draw(im)
    d.polygon([(0, 700), (500, 520), (900, 690), (1300, 470), (1700, 680), (WW, 560), (WW, WH), (0, WH)], fill=(100, 140, 190))
    d.rectangle([0, FEET - 30, WW, WH], fill=(205, 232, 250))
    for k in range(22):                                                     # distant endless queue
        s = 1 - ((k - (t * .4) % 1) / 22); s = clamp(s); x = 1750 - k * 68; h = 150 * (1 - k / 30)
        d.rounded_rectangle([x - 16, FEET - 30 - h, x + 16, FEET - 30], 8, fill=(25, 40, 90)); d.ellipse([x - 12, FEET - 30 - h - 26, x + 12, FEET - 30 - h], fill=(25, 40, 90))
    snow(d, t); return im, d

def bg_z(t, v):
    im = G[Z].copy(); d = ImageDraw.Draw(im); cols = [(190, 255, 60), (255, 255, 255), (60, 235, 255), (255, 80, 225)]
    for k in range(5):
        d.line([(x, 500 + (170 - 20 * k) * math.sin(x / (240 + 30 * k) + t * (3 + k) + k)) for x in range(0, WW + 40, 40)], fill=cols[k % 4], width=30)
    d.rectangle([0, FEET - 10, WW, WH], fill=(70, 20, 130))
    for k in range(10):
        px = (k * 230 + t * (120 + 40 * k)) % (WW + 200) - 100; py = 150 + 90 * math.sin(t * 4 + k); r = 24 + 14 * (k % 3); d.ellipse([px - r, py - r, px + r, py + r], fill=cols[k % 4])
    return im, d

def bg_dawn(t, v, u=None):
    u = clamp((t - 87.4) / 14) if u is None else u
    im = G["dawn"].copy(); d = ImageDraw.Draw(im); sy = 760 - 380 * u
    for k in range(8, 0, -1): d.ellipse([960 - 130 - k * 40, sy - 130 - k * 40, 960 + 130 + k * 40, sy + 130 + k * 40], fill=(255, 210 - k * 3, 150 - k * 6))
    d.ellipse([960 - 130, sy - 130, 960 + 130, sy + 130], fill=(255, 245, 220))
    d.polygon([(0, WH), (0, 760), (350, 560), (700, 780), (1100, 620), (1500, 800), (WW, 650), (WW, WH)], fill=(20, 40, 90))
    d.rectangle([0, FEET - 20, WW, WH], fill=(25, 45, 95)); snow(d, t, 90, 60); return im, d

def bg_card(t, gen, v):
    a, b = COL[gen]; im = G["card"].copy(); d = ImageDraw.Draw(im)
    for k in range(-6, 14):                                                    # diagonal stripes
        x = k * 220 + (t * 120) % 220; d.polygon([(x, 0), (x + 90, 0), (x - 300 + 90, WH), (x - 300, WH)], fill=tuple(int(c * .45) for c in a))
    d.ellipse([560, 260, 1360, 1060], fill=tuple(int(c * .6) for c in b))
    return im, d
BG = {BUB: bg_bub, MIL: bg_mil, ICE: bg_ice, Z: bg_z}

# ---------------------------------------------------------------- world props (in scene)
def wprops(d, name, t, lt, T, gx, h, l):
    f = font("lat", 60)
    if name == "crowd": pass
    elif name == "card":
        a = 1 - clamp(lt / T); c = tuple(int(x * a) for x in (255, 250, 230))
        d.rounded_rectangle([gx + 260, 380 + 30 * math.sin(t * 3), gx + 620, 570 + 30 * math.sin(t * 3)], 18, fill=c)
        d.text((gx + 440, 440 + 30 * math.sin(t * 3)), "部長", font=font("jp", 90), fill=tuple(int(x * a) for x in (60, 20, 70)), anchor="mm")
    elif name == "code":
        for k in range(30):
            px = 60 + k * 62; py = (k * 137 + t * 400) % WH
            d.text((px, py), "01"[(k + int(t * 8)) % 2], font=font("lat", 50), fill=(255, 210, 90))
    elif name == "rays":
        for k in range(14):
            a0 = k * .45 + t * .3; d.polygon([(gx, FEET - h * .6), (gx + 1500 * math.cos(a0), FEET - h * .6 + 1500 * math.sin(a0)), (gx + 1500 * math.cos(a0 + .12), FEET - h * .6 + 1500 * math.sin(a0 + .12))], fill=(255, 225, 140))
    elif name == "ques":
        for k in range(8): d.text((gx - 500 + k * 140, 230 + 60 * math.sin(t * 3 + k) + (k % 3) * 100), "?", font=font("lat", 130), fill=(140, 150, 255))
    elif name == "menu":
        for k in range(3):
            y0 = 380 + k * 110; sel = k == int(t * 2) % 3
            d.rounded_rectangle([gx + 240, y0, gx + 640, y0 + 90], 16, fill=(40, 230, 215) if sel else (30, 60, 90)); d.text((gx + 440, y0 + 45), ["PRESET A", "PRESET B", "PRESET C"][k], font=font("lat", 40), fill=(10, 20, 40) if sel else (140, 200, 220), anchor="mm")
    elif name == "loading":
        u = (t % 2.2) / 2.2; bx = gx - 300; by = FEET - h - 120
        d.rounded_rectangle([bx, by, bx + 600, by + 50], 25, outline=(255, 255, 255), width=6); d.rounded_rectangle([bx + 6, by + 6, bx + 6 + 588 * u, by + 44], 20, fill=(40, 230, 215))
        d.text((gx, by - 50), f"{int(u * 99)}%", font=font("lat", 60), fill=(255, 255, 255), anchor="mm")
    elif name == "frost":
        u = clamp(lt / (T * .9))
        for k in range(16):
            x0 = k * 130; hh = 200 * u * (.5 + .5 * math.sin(k * 2.3)) + 30 * u
            d.polygon([(x0, WH), (x0 + 60, WH - hh * 2.2), (x0 + 120, WH)], fill=(200, 235, 255))
            d.polygon([(x0, 0), (x0 + 50, hh * 1.4), (x0 + 100, 0)], fill=(200, 235, 255))
    elif name == "chair":
        cx = 1500; d.rectangle([cx, FEET - 190, cx + 40, FEET - 20], fill=(255, 250, 220)); d.rectangle([cx, FEET - 110, cx + 150, FEET - 84], fill=(255, 250, 220)); d.rectangle([cx + 120, FEET - 84, cx + 140, FEET - 20], fill=(255, 250, 220))
    elif name == "number":
        d.rounded_rectangle([gx + 220, 260, gx + 720, 420], 14, fill=(20, 25, 50), outline=(255, 90, 90), width=6)
        d.text((gx + 470, 340), f"No.{1042 - int(t * 2) % 10:04d}", font=font("tag", 110), fill=(255, 90, 90), anchor="mm")
    elif name == "tag":
        a = 1 - clamp((lt - .3) / (T * .8)); c = tuple(int(x * a) for x in (255, 255, 255))
        d.rounded_rectangle([gx - 60, 540, gx + 60, 620], 8, fill=c, outline=(20, 20, 40), width=3)
        d.text((gx, 580), "???", font=font("lat", 34), fill=tuple(int(x * a) for x in (60, 60, 90)), anchor="mm")
    elif name == "poster":
        d.rounded_rectangle([gx + 240, 180, gx + 660, 620], 12, fill=(255, 230, 120), outline=(60, 40, 10), width=6)
        d.text((gx + 450, 400), "DREAM", font=font("lat", 90), fill=(200, 60, 40), anchor="mm")
    elif name == "calendar":
        for k in range(14):
            x0 = (k * 210 - t * 500) % (WW + 260) - 200; y0 = 180 + (k % 5) * 110
            d.rounded_rectangle([x0, y0, x0 + 190, y0 + 80], 10, fill=(255, 255, 255) if k % 3 else (255, 120, 120))
    elif name == "veil":
        for k in range(14):
            x = 1500 + 32 * k + 26 * math.sin(t * 2 + k); d.line([x, 0, x + 40 * math.sin(t + k), WH], fill=(255, 255, 255), width=16 - k)
    elif name == "door":
        d.rectangle([1450, FEET - 470, 1730, FEET - 20], fill=(30, 200, 110), outline=(255, 255, 255), width=8)
        d.text((1590, FEET - 520), "EXIT", font=font("lat", 70), fill=(255, 255, 255), anchor="mm")
    elif name == "toggle":
        on = int(t * 2) % 2 == 0
        d.rounded_rectangle([gx + 260, 320, gx + 620, 440], 60, fill=(190, 255, 60) if on else (110, 60, 160)); kx = gx + 560 if on else gx + 320
        d.ellipse([kx - 50, 330, kx + 50, 430], fill=(255, 255, 255))
    elif name == "clock":
        u = .35 + .05 * math.sin(t); cx, cy, r = 1400, 380, 200
        d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * u, fill=(255, 255, 255), width=24)
    elif name == "dawnfx":
        for k in range(10): d.line([(60 + k * 200, WH), (60 + k * 200 + 40, FEET - 10)], fill=(255, 235, 180), width=3)

def gen_pos(i, v):
    """character x for line i and view v"""
    return (620 if i % 2 == 0 else 1300) if v == 0 else 960

def scene_line(i, t, v):
    return _scene_line(i, t, v)

def _scene_line(i, t, v):
    l = LINES[i]; lt = t - l["t"]; T = LINE_END[i] - l["t"]; w = l["w"]
    h = 430 if v == 0 else 560
    if w == SPL:
        im = bg_ice(t, v)[0] if i % 2 == 0 else bg_z(t, v)[0]; d = ImageDraw.Draw(im)
        m = Image.new("L", (WW, WH), 0); ImageDraw.Draw(m).polygon([(0, 0), (1250, 0), (700, WH), (0, WH)], fill=255)
        b = (bg_bub if i % 2 == 0 else bg_mil)(t, v)[0]; im.paste(b, (0, 0), m)
        ImageDraw.Draw(im).line([(1250, 0), (700, WH)], fill=(255, 255, 255), width=12)
        hs = 400 if v == 0 else 520
        a, b2 = (BUB, MIL) if i % 2 else (ICE, Z)
        if i == 23:
            for gi, (gen, x) in enumerate(zip([BUB, MIL, ICE, Z], [640, 830, 1090, 1290])):
                place(im, gen, "fist" if gen == ICE else "dance" if gen == Z else "walk" if gen == MIL else "celebrate", t + gi, 360 if gen != ICE else 420, x, FEET, gen in (ICE, Z))
            return im
        place(im, ICE if i % 2 == 0 else BUB, l["act"] if i % 2 == 0 else "celebrate", t, hs, 500, FEET, False, "case" if i % 2 == 0 else "glass", "sad" if i % 2 == 0 else "smile")
        place(im, Z if i % 2 == 0 else MIL, "dance" if i % 2 == 0 else "shrug", t, hs, 1400, FEET, True, "phone", "smile" if i % 2 == 0 else "meh")
        return im
    if w == ICE and l["wp"] == "dawnfx": im, d = bg_dawn(t, v)
    else: im, d = BG[w](t, v)
    gx = gen_pos(i, v)
    wprops(d, l["wp"], t, lt, T, gx - 450 if i % 2 else gx, h, l)
    if l["wp"] == "crowd" and v == 0:
        for k, (dx, hs, ph) in enumerate([(-420, 400, 1.3), (420, 420, 2.2)]):
            place(im, BUB, "celebrate", t + ph, hs, gx + dx, FEET - 40, k == 1, "glass" if k else "none")
    if l["w"] == BUB and l["wp"] == "card": al = 1 - .8 * clamp(lt / T)
    else: al = 1.0
    x = gx + (lt * 260 if l["act"] == "walk" and v == 0 else 0) * 0
    place(im, w, l["act"], t, h, x, FEET + (30 if v else 0), i % 2 == 1 and l["act"] not in ("run",), l["hp"], l["mood"], al)
    if l["wp"] == "frost": wprops(ImageDraw.Draw(im), "frost", t, lt, T, gx, h, l)
    return im

def scene_generic(kind, t, v, gen=None, lt=0):
    if kind == "card":
        im, d = bg_card(t, gen, v); a, b = COL[gen]
        act = {BUB: "celebrate", MIL: "shrug", ICE: "fist", Z: "dance"}[gen]
        hp = {BUB: "glass", MIL: "phone", ICE: "case", Z: "phone"}[gen]
        e = ease(lt / .35); xoff = (1 - e) * 900 * (1 if gen in (BUB, ICE) else -1)
        place(im, gen, act, t, 640 if v == 0 else 720, 960 + xoff, FEET + 30, gen in (MIL, Z), hp, "sad" if gen == ICE else "smile")
        return im
    if kind == "lineup":
        im, d = bg_ice(t, v); im = Image.blend(im, Image.new("RGB", (WW, WH), (10, 15, 40)), .35)
        for gi, (g_, x, h) in enumerate([(BUB, 640, 470), (MIL, 830, 470), (ICE, 1030, 560), (Z, 1260, 470)]):
            place(im, g_, {BUB: "celebrate", MIL: "shrug", ICE: "fist", Z: "dance"}[g_], t + gi, h, x, FEET, g_ in (ICE, Z),
                  {BUB: "glass", MIL: "phone", ICE: "case", Z: "phone"}[g_], "sad" if g_ == ICE else "smile")
        return im
    if kind == "hero":
        im, d = bg_dawn(t, v, .35); place(im, ICE, "run" if v == 0 else "fist", t, 560 if v == 0 else 700, 960, FEET, False, "case", "smile"); return im
    if kind == "glitch":
        im = Image.new("RGB", (WW, WH), (8, 8, 18)); d = ImageDraw.Draw(im)
        for k in range(40): d.line([(0, (k * 97 + t * 900) % WH), (WW, (k * 97 + t * 900) % WH)], fill=(30, 40, 90), width=2)
        g_ = [BUB, MIL, ICE, Z][int(t * 2 / .513) % 4]
        place(im, g_, "idle", t, 640, 960, FEET + 30); return im
    return bg_ice(t, v)[0]

def scene_for(t, shot):
    v = shot % 2; i = line_at(t)
    if i is not None: return scene_line(i, t, v)
    if t < 2.1: return scene_generic("glitch", t, v)
    if t < 5.2: return scene_generic("hero", t, v)
    if t < 13.4:
        k = max(j for j, s in enumerate(INTRO_CARDS) if s <= t); return scene_generic("card", t, v, GEN[k][2], t - INTRO_CARDS[k])
    if t < 18.2: return scene_generic("lineup", t, v)
    if t < 87.4:
        if t < 79.9: return scene_generic("lineup", t, v)
        if t < 83.9:
            k = shot % 4; return scene_generic("card", t, v, GEN[k][2], (t - CUTS[shot]) + .2)
        return scene_generic("hero", t, v)
    return scene_generic("hero", t, v)

def focus_for(t, shot):
    v = shot % 2; i = line_at(t)
    if i is not None and LINES[i]["w"] != SPL:
        x = gen_pos(i, v); sgn = 1 if i % 2 == 0 else -1
        return ((x + 200 * sgn, FEET - 230) if v == 0 else (960 + 150 * sgn, FEET - 440)), (1.0 if v == 0 else 1.1)
    if i is None and 5.2 <= t < 13.4: return (960, FEET - 400), .9
    return (960, FEET - 390), .95

# ---------------------------------------------------------------- camera
def camera(im, cam, seed, u, t, focus=(WW / 2, WH / 2), zmul=1.0):
    r = np.random.default_rng(seed); sgn = 1 if r.random() < .5 else -1
    z, th, cx, cy = 1.15, 0.0, focus[0], focus[1]; e = ease(u)
    if cam == "push": z = 1.0 + .3 * e
    elif cam == "pull": z = 1.35 - .35 * e
    elif cam == "panL": z = 1.1; cx += (140 - 280 * e) * sgn
    elif cam == "panR": z = 1.12; cx += (-140 + 280 * e) * sgn
    elif cam == "tiltU": z = 1.15; cy += 90 - 180 * e
    elif cam == "roll": z = 1.2; th = math.radians((-9 + 18 * u) * sgn)
    elif cam == "whip": z = 1.1; cx += 600 * (1 - e) * sgn; th = math.radians(6 * (1 - e) * sgn)
    elif cam == "dutch": z = 1.15 + .1 * u; th = math.radians(11 * sgn)
    elif cam == "spin": z = 1.25 + .1 * u; th = math.radians(28 * (1 - e) * sgn)
    z *= zmul * (1 + .05 * punch(t) + .04 * af(bass, t))
    cx += math.sin(t * 27) * 5 * (.3 + af(rms, t)); cy += math.cos(t * 31) * 5 * (.3 + af(rms, t))
    c, s = math.cos(th), math.sin(th)
    hw = (W * c + H * abs(s)) / (2 * z); hh = (W * abs(s) + H * c) / (2 * z)
    if hw > WW / 2 or hh > WH / 2:
        z *= max(hw / (WW / 2), hh / (WH / 2)); hw = (W * c + H * abs(s)) / (2 * z); hh = (W * abs(s) + H * c) / (2 * z)
    cx = min(max(cx, hw), WW - hw); cy = min(max(cy, hh), WH - hh)
    a, b = c / z, -s / z; d_, e_ = s / z, c / z
    return im.transform((W, H), Image.AFFINE, (a, b, cx - W / 2 * a - H / 2 * b, d_, e_, cy - W / 2 * d_ - H / 2 * e_), Image.BILINEAR)

# ---------------------------------------------------------------- kinetic typography
@lru_cache(maxsize=6000)
def glyph(ch, fk, size, fill, sw, sc):
    f = font(fk, size); pad = sw + 6; adv = f.getlength(ch)
    tile = Image.new("RGBA", (int(adv) + 2 * pad + 4, int(size * 1.5) + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((tile.width / 2, tile.height / 2), ch, font=f, fill=fill, stroke_width=sw, stroke_fill=sc, anchor="mm")
    return tile, adv

def put(ov, ch, fk, size, cx, cy, sx=1.0, sy=1.0, rot=0.0, alpha=1.0, fill=(255, 255, 255), sw=8, sc=(15, 15, 40), shadow=True):
    if alpha <= .01 or sx == 0: return
    tile, _ = glyph(ch, fk, int(size), fill + (255,), sw, sc + (255,))
    if sx != 1 or sy != 1: tile = tile.resize((max(1, int(tile.width * abs(sx))), max(1, int(tile.height * sy))), Image.BILINEAR)
    if rot: tile = tile.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1: tile = tile.copy(); tile.putalpha(tile.getchannel("A").point(lambda v: int(v * alpha)))
    x, y = int(cx - tile.width / 2), int(cy - tile.height / 2)
    if shadow:
        sh = Image.new("RGBA", tile.size, (0, 0, 0, 0)); sh.putalpha(tile.getchannel("A").point(lambda v: int(v * .45))); ov.paste(sh, (x + 8, y + 10), sh)
    ov.paste(tile, (x, y), tile)

def kinetic(ov, word, style, lt, T, cx, cy, size, acc, acc2, t):
    fk = "jp" if any(ord(c) > 255 for c in word) else "lat"
    f = font(fk, size); sp = size * .04
    advs = [f.getlength(c) for c in word]; total = sum(advs) + sp * (len(word) - 1)
    maxw = 2 * min(cx, W - cx) - 40
    if total > maxw: size = size * maxw / total; f = font(fk, size); advs = [f.getlength(c) for c in word]; sp = size * .04; total = sum(advs) + sp * (len(word) - 1)
    d = ImageDraw.Draw(ov); n = len(word); x = cx - total / 2
    outa = 1 - clamp((lt - (T - .22)) / .22)
    ring_t = lt - .18
    centers = []
    for c_, a_ in zip(word, advs): centers.append(x + a_ / 2); x += a_ + sp
    if style == "node":
        for a, b in zip(centers, centers[1:]): d.line([(a, cy), (b, cy)], fill=acc + (int(220 * outa),), width=8)
    if style == "lock":
        e = ease(lt / .3); m = (1 - e) * 120 + 20; L_ = size * .5
        for sx_, sy_ in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            px, py = cx + sx_ * (total / 2 + m), cy + sy_ * (size * .45 + m * .5)
            d.line([(px, py), (px - sx_ * L_ * .6, py)], fill=acc + (int(255 * outa),), width=10); d.line([(px, py), (px, py - sy_ * L_ * .6)], fill=acc + (int(255 * outa),), width=10)
    if style in ("slam", "lock") and 0 < ring_t < .5:
        r = 60 + ring_t * 1800; d.ellipse([cx - r, cy - r * .35, cx + r, cy + r * .35], outline=acc + (int(200 * (1 - ring_t / .5)),), width=8)
    if style == "slide" and lt < .5:
        for k in range(9):
            yy = cy - size * .5 + k * size / 8; l0 = cx - total / 2 - 800 + 1500 * ease(lt / .3)
            d.line([(l0, yy), (l0 + 700 * (1 - lt / .5), yy)], fill=acc + (int(160 * (1 - lt / .5)),), width=6)
    if style == "fill":
        u = clamp(lt / (T * .8))
        for i_, (c_, cxx) in enumerate(zip(word, centers)):
            put(ov, c_, fk, size, cxx, cy, alpha=outa, fill=(25, 25, 55), sw=8, sc=acc, shadow=False)
        lay = Image.new("RGBA", ov.size, (0, 0, 0, 0))
        for c_, cxx in zip(word, centers): put(lay, c_, fk, size, cxx, cy, alpha=outa, fill=acc, sw=8, sc=(255, 255, 255), shadow=False)
        m = Image.new("L", ov.size, 0); ImageDraw.Draw(m).rectangle([cx - total / 2 - 20, 0, cx - total / 2 - 20 + (total + 40) * u, ov.height], fill=255)
        lay.putalpha(Image.composite(lay.getchannel("A"), Image.new("L", ov.size, 0), m)); ov.paste(lay, (0, 0), lay); return
    for i_, (c_, cxx) in enumerate(zip(word, centers)):
        li = lt - i_ * (.05 if style not in ("type", "vanish", "node") else .1 if style != "type" else .09)
        e = ease(li / .2); a = e * outa; sx = sy = 1.0; dx = dy = 0.0; rot = 0.0; col = (255, 255, 255); sc = acc; sw = 9
        if style == "slam": sx = sy = 1 + (1 - e) * 2.4; dy = -(1 - e) * 320; rot = (1 - e) * 20 * (1 if i_ % 2 else -1); dx = math.sin(li * 90) * 10 * clamp(1 - (li - .2) * 4) * (li > .2)
        elif style == "fade":
            a = e * (1 - clamp((lt - T * .35 - i_ * .08) / .5)); dy = -clamp((lt - T * .35 - i_ * .08) / .5) * 90; sx = sy = 1 + (1 - e) * .5; col = (255, 255, 255) if lt < T * .35 else acc
        elif style == "vanish":
            a = e * (1 if lt < T * .35 + i_ * .2 else 0) * outa; sx = sy = 1 + (1 - e) * .8
            if 0 < lt - (T * .35 + i_ * .2) < .2:
                for q in range(7): d.ellipse([cxx + 60 * math.cos(q * .9) * (lt - T * .35 - i_ * .2) * 6 - 6, cy + 60 * math.sin(q * .9) * (lt - T * .35 - i_ * .2) * 6 - 6] + [cxx + 60 * math.cos(q * .9) * (lt - T * .35 - i_ * .2) * 6 + 6, cy + 60 * math.sin(q * .9) * (lt - T * .35 - i_ * .2) * 6 + 6], fill=acc + (200,)); a = .0 if False else a
        elif style == "type":
            a = (1 if li > 0 else 0) * outa; col = acc if li > 0 else col; dx = (np.random.default_rng(int(t * 12) + i_).random() - .5) * 8 * (punch(t) > .5)
            if 0 < li < .09: sx = 1.3
        elif style == "lock": sx = sy = 1 + (1 - e) * 1.2; col = (255, 255, 255)
        elif style == "slide": dx = -(1 - e) * 900 + lt * 25; sx = 1 + (1 - e) * .8; sy = 1 - (1 - e) * .15
        elif style == "freeze":
            k = clamp(lt / (T * .5)); col = tuple(int(255 * (1 - k) + c * k) for c in (150, 215, 255)); dx = math.sin(lt * 70) * 6 * (1 - k); sx = sy = 1 + (1 - e) * .8
        elif style == "wave": dy = math.sin(lt * 9 - i_ * .7) * size * .14; sx = sy = e * .5 + .5
        elif style == "spin": sx = math.cos(clamp(1 - li / .5) * math.pi * 2 - (0 if li > .5 else 0)) if li < .5 else 1; sy = 1; col = (255, 255, 255)
        elif style == "bounce": dy = -abs(math.sin(lt * 7 - i_ * .5)) * size * .16; sx = sy = e * .4 + .6
        elif style == "glitch":
            g = np.random.default_rng(int(t * 15) + i_); dx = (g.random() - .5) * 26 * (punch(t) > .3); dy = (g.random() - .5) * 12 * (punch(t) > .3)
            if punch(t) > .35: put(ov, c_, fk, size, cxx + dx - 8, cy + dy, alpha=a * .8, fill=(255, 60, 90), sw=0, shadow=False); put(ov, c_, fk, size, cxx + dx + 8, cy + dy, alpha=a * .8, fill=(60, 230, 255), sw=0, shadow=False)
        elif style == "node":
            if li > 0: d.ellipse([cxx - size * .55, cy - size * .55, cxx + size * .55, cy + size * .55], fill=acc + (int(230 * outa),), outline=(255, 255, 255, int(255 * outa)), width=5)
            col = (10, 15, 40); sc = (255, 255, 255); sw = 0
        put(ov, c_, fk, size, cxx + dx, cy + dy, sx, sy, rot, a, col, sw, (255, 255, 255) if style == "node" else (15, 25, 70) if style == "freeze" else (15, 15, 40))
    if style in ("slam", "lock") and lt < .25:  # accent under-bar
        d.rectangle([cx - total / 2, cy + size * .55, cx - total / 2 + total * ease(lt / .25), cy + size * .55 + 12], fill=acc + (int(255 * outa),))

TAG_FONT = "tag"
def gen_tag(ov, gen, lt):
    for jp, en, w in GEN:
        if w == gen:
            a = ease(lt / .3); x = 40 - (1 - a) * 300; d = ImageDraw.Draw(ov); col = COL[w][0]
            d.polygon([(x, 30), (x + 330, 30), (x + 300, 100), (x, 100)], fill=(10, 12, 30, 200)); d.rectangle([x, 30, x + 10, 100], fill=col + (255,))
            d.text((x + 28, 50), en, font=font("tag", 46), fill=col + (255,), anchor="lm"); d.text((x + 28, 84), jp, font=font("jpb", 20), fill=(255, 255, 255, 230), anchor="lm")

def title_glitch(ov, txt, cx, cy, size, t, alpha=1.0, acc=(175, 225, 255)):
    f = font("jp", size); advs = [f.getlength(c) for c in txt]; total = sum(advs) + size * .04 * (len(txt) - 1); x = cx - total / 2
    for c, a in zip(txt, advs):
        put(ov, c, "jp", size, x + a / 2, cy, alpha=alpha, fill=(255, 255, 255), sw=int(size * .07), sc=(20, 60, 150)); x += a + size * .04

# ---------------------------------------------------------------- overlay
def overlay(t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); p = punch(t); i = line_at(t); cx = W // 2
    if i is not None:
        l = LINES[i]; lt = t - l["t"]; T = LINE_END[i] - l["t"]; acc, acc2 = COL[l["w"]]
        pos = (860, 150) if i % 2 == 0 else (420, 150)
        if len(l["big"]) > 5: pos = (cx, 150)
        if l["w"] == SPL and i == 23: pos = (cx, 160)
        sz = 230 if len(l["big"]) <= 6 else 165
        if l["big"] == "非対称戦": sz = 190
        kinetic(ov, l["big"], l["st"], lt, T, pos[0], pos[1], sz * (1 + .03 * p), acc, acc2, t)
        gen_tag(ov, l["w"] if l["w"] != SPL else ICE, lt)
        return ov
    if t < 2.1:
        if int(t * 8) % 2 == 0 or t < .1: d.rectangle([0, H // 2 - 2, W, H // 2 + 2], fill=(255, 255, 255, 200))
        for k in range(4):
            j = int((t - .05) / .513) % 4
            if k == j: put(ov, GEN[k][1][0], "tag", 0, 0, 0, alpha=0) if False else None
        g = GEN[int(t / .513) % 4]; col = COL[g[2]][0]
        put(ov, "!", "lat", 1, 0, 0, alpha=0) if False else None
        kinetic(ov, g[1], "glitch", t % .513, .513, cx, H // 2, 100, col, col, t)
    elif t < 5.2:
        lt = t - 2.1; n = int(clamp(lt / .513 + 1e-3, 0, 6)) if lt < 3.1 else 6
        f = font("jp", 150); size = 150; advs = [f.getlength(c) for c in TITLE]; total = sum(advs) + size * .04 * 5; x = cx - total / 2
        for k, (c, a) in enumerate(zip(TITLE, advs)):
            li = lt - k * .513
            if li >= 0:
                e = ease(li / .12); s = 1 + (1 - e) * 2.5; shake = math.sin(li * 90) * 14 * clamp(1 - li * 5)
                put(ov, c, "jp", size, x + a / 2 + shake, H // 2 - 20 + shake * .5, s, s, (1 - e) * 15, e, (255, 255, 255), 11, (20, 60, 150))
                if li < .3:
                    r = 40 + li * 1800; d.ellipse([x + a / 2 - r, H // 2 - 20 - r * .4, x + a / 2 + r, H // 2 - 20 + r * .4], outline=(175, 225, 255, int(230 * (1 - li / .3))), width=6)
            x += a + size * .04
        if lt > 2.9:
            sub = "ASYMMETRIC  WARFARE"; kinetic(ov, sub, "type", lt - 2.9, 2.0, cx, H // 2 + 110, 44, (175, 225, 255), (255, 255, 255), t)
    elif t < 13.4:
        k = max(j for j, s in enumerate(INTRO_CARDS) if s <= t); lt = t - INTRO_CARDS[k]; jp, en, w = GEN[k]; a, b = COL[w]
        e = ease(lt / .3); x = 60 - (1 - e) * 700
        d.polygon([(x, 470), (x + 760, 470), (x + 700, 600), (x, 600)], fill=(10, 12, 30, 225)); d.polygon([(x, 470), (x + 24, 470), (x + 24, 600), (x, 600)], fill=a + (255,))
        d.text((x + 48, 512), en, font=font("lat", 66), fill=a + (255,), anchor="lm"); d.text((x + 50, 574), jp, font=font("jp", 40), fill=(255, 255, 255, 255), anchor="lm")
        yrs = {BUB: "1986 - 1991", MIL: "1981 - 1996", ICE: "1993 - 2004", Z: "1997 -"}[w]
        d.text((x + 560, 574), yrs, font=font("tag", 44), fill=b + (255,), anchor="lm")
        put(ov, "VS", "lat", 1, 0, 0, alpha=0) if False else None
        if k == 2 and lt > .5: d.text((W - 40, 60), "PROTAGONIST", font=font("tag", 50), fill=(255, 255, 255, 255), anchor="rm")
    elif t < 15.4:
        kinetic(ov, "4 GENERATIONS", "slide", t - 13.4, 2.0, cx, 230, 120, (255, 255, 255), (255, 255, 255), t)
        kinetic(ov, "1 SOCIETY", "slam", t - 13.9, 1.5, cx, 400, 130, (175, 225, 255), (255, 255, 255), t)
    elif t < 18.2:
        lt = t - 15.4; title_glitch(ov, TITLE, cx, H // 2 - 10, int(170 * (1 + .04 * p)), t, ease(lt / .15))
        kinetic(ov, "ROUND 1", "slide", lt, 2.8, cx, H // 2 + 130, 60, (255, 90, 90), (255, 255, 255), t)
        if t > 17.7: d.rectangle([0, 0, W, H], fill=(255, 255, 255, int(255 * clamp((t - 17.7) / .5))))
    elif t < 75.9:
        pass
    elif t < 87.4:
        lt = t - 75.9
        if lt < 4: kinetic(ov, "SAME SOCIETY", "slide", lt, 4, cx, 170, 110, (255, 255, 255), (255, 255, 255), t)
        elif lt < 8:
            kinetic(ov, "DIFFERENT STAGE", "slam", lt - 4, 4, cx, 170, 100, (175, 225, 255), (255, 255, 255), t)
        else: title_glitch(ov, TITLE, cx, H // 2 - 10, int(160 * (1 + .04 * p)), t, ease((lt - 8) / .15))
        if lt >= 4 and lt < 8: gen_tag(ov, [BUB, MIL, ICE, Z][int((t - 79.9) / .513 / 2) % 4], (t - 79.9) % 1.03)
    elif t >= 101.0:
        a = clamp((t - 101) / 1.2); title_glitch(ov, TITLE, cx, H // 2 - 20, 170, t, a)
        put(ov, "p", "lat", 1, 0, 0, alpha=0) if False else None
        d.text((cx, H // 2 + 110), "produced by okita", font=font("lat", 54), fill=(200, 230, 255, int(255 * a)), anchor="mm", stroke_width=5, stroke_fill=(0, 0, 40, int(255 * a)))
    return ov

# ---------------------------------------------------------------- frame
VIG = None
def vignette():
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W]; dd = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        VIG = np.clip(1 - .5 * dd ** 2.2, .4, 1)[..., None].astype(np.float32)
    return VIG

def frame(i):
    t = i / FPS
    s = max(np.searchsorted(CUTS, t, side="right") - 1, 0); s_end = CUTS[s + 1] if s + 1 < len(CUTS) else DUR
    u = (t - CUTS[s]) / max(s_end - CUTS[s], 1e-3)
    cam, seed = cam_for(s)
    if s_end - CUTS[s] < .9 and cam in ("whip", "pull"): cam = "push"
    fc, zm = focus_for(t, s)
    im = camera(scene_for(t, s), cam, seed, u, t, fc, zm)
    arr = np.asarray(im, np.float32); on = af(onset, t)
    if on > .55:
        sh = int(10 * on); arr = np.stack([np.roll(arr[..., 0], sh, 1), arr[..., 1], np.roll(arr[..., 2], -sh, 1)], -1)
    arr = arr * vignette() + 45 * punch(t) * (1 if 67.7 <= t < 76 or on > .8 else .3) + 110 * max(0, 1 - (t - CUTS[s]) / .08)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    im = Image.alpha_composite(im, overlay(t)).convert("RGB")
    if 67.7 <= t < 75.9:
        k = min(int((t - 67.7) / 2.05), 3); fade = max(0, 1 - (t - (67.7 + k * 2.05)) / .15)
        if fade: im = Image.blend(im, Image.new("RGB", (W, H), COL[[BUB, MIL, ICE, Z][k]][0]), .5 * fade)
    return im

def render(i): return frame(i).tobytes()

if __name__ == "__main__":
    if PREVIEW:
        for t in PREVIEW: frame(int(t * FPS)).save(f"{OUT}_{t:g}.png")
        sys.exit()
    n = int(DUR * FPS)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-i", AUDIO, "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, b in enumerate(pool.imap(render, range(n), chunksize=8)):
            ff.stdin.write(b)
            if k % 300 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
