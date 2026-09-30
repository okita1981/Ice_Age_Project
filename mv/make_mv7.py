"""MV generator v3 for 「世代非対称戦」: generation characters, kinetic English rhyme words, fighting-game intro.

usage: python mv/make_mv3.py <audio.mp3> <out.mp4> [--preview t1,t2,...]
"""
import sys, math, subprocess
from functools import lru_cache
import numpy as np, librosa, imageio_ffmpeg
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

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
      "jpb": "/usr/share/fonts/opentype/mplus/Mplus1-Bold.otf",
      "jpt": "/usr/share/fonts/opentype/mplus/Mplus1-Thin.otf", "jpl": "/usr/share/fonts/opentype/mplus/Mplus1-Light.otf",
      "latl": "/usr/share/fonts/opentype/montserrat/Montserrat-Light.otf",
      "latR": "/usr/share/fonts/opentype/montserrat/Montserrat-Regular.otf", "latL": "/usr/share/fonts/opentype/montserrat/Montserrat-Light.otf",
      "latX": "/usr/share/fonts/opentype/montserrat/Montserrat-ExtraLight.otf", "jpr": "/usr/share/fonts/opentype/mplus/Mplus1-Regular.otf"}
@lru_cache(maxsize=None)
def font(k, sz): return ImageFont.truetype(FP[k], int(sz))

BUB, MIL, ICE, Z, SPL = "bub", "mil", "ice", "z", "split"
COL = {BUB: ((255, 196, 60), (255, 60, 160)), MIL: ((40, 230, 215), (140, 150, 255)),
       ICE: ((175, 225, 255), (255, 255, 255)), Z: ((190, 255, 60), (255, 80, 225)), SPL: ((255, 255, 255), (255, 90, 90))}
GEN = [("バブル世代", "BUBBLE ERA", BUB), ("ミレニアル世代", "MILLENNIAL", MIL), ("氷河期世代", "ICE AGE", ICE), ("Z世代", "GEN Z", Z)]

# (line start, rhyme-word onset, word, gen, style, actions, hand prop, world prop, mood, weight B/R/L/X)
def L(t, w0, big, w, st, acts, hp="none", wp="none", mood="smile", wt="R"): return dict(t=t, w0=w0, big=big, w=w, st=st, acts=acts, hp=hp, wp=wp, mood=mood, wt=wt)
LINES = [
    L(18.20, 19.80, "RAID", BUB, "slam", ("celebrate", "disco", "toast"), "glass", "crowd", wt="R"),
    L(20.28, 21.82, "FADE", BUB, "fade", ("strut", "phoneear", "toast"), "none", "card", wt="X"),
    L(22.32, 23.88, "CODE", BUB, "type", ("laugh", "toast", "disco"), "glass", "code", wt="R"),
    L(24.36, 25.86, "HOLD", BUB, "lock", ("fist", "present", "toast"), "trophy", "rays", wt="B"),
    L(26.38, 27.98, "ROAD", MIL, "slide", ("walk", "present", "type"), "tab", "none", "cool", "L"),
    L(28.32, 30.12, "NODE", MIL, "node", ("watch", "type", "cool"), "tab", "ques", "cool", "R"),
    L(30.44, 32.22, "MODE", MIL, "spin", ("type", "cross", "cool"), "tab", "menu", "cool", "X"),
    L(32.54, 34.30, "LOAD", MIL, "fill", ("watch", "cross", "present"), "tab", "loading", "cool", "R"),
    L(34.64, 36.14, "PHASE", ICE, "freeze", ("slump", "wipe", "push"), "case", "frost", "grit", "B"),
    L(36.60, 38.18, "RACE", ICE, "slide", ("run", "stumble", "run"), "case", "chair", "grit", "L"),
    L(38.58, 40.32, "CASE", ICE, "lock", ("bow", "idle", "wipe"), "paper", "number", "grit", "R"),
    L(40.62, 42.30, "NAME", ICE, "vanish", ("slump", "reach", "idle"), "none", "tag", "grit", "X"),
    L(42.74, 44.34, "DAYS", ICE, "bounce", ("reach", "slump", "wipe"), "case", "poster", "grit", "L"),
    L(44.90, 46.48, "PACE", ICE, "slide", ("run", "stumble", "push"), "case", "calendar", "grit", "R"),
    L(46.88, 48.54, "VEIL", ICE, "fade", ("run", "push", "reach"), "case", "veil", "grit", "X"),
    L(48.90, 50.72, "GAME", ICE, "spin", ("idle", "wipe", "bow"), "book", "none", "grit", "L"),
    L(51.18, 52.70, "SWAY", Z, "wave", ("dance", "gaming", "skate"), "game", "none", "smile", "R"),
    L(53.04, 54.52, "OK", Z, "bounce", ("jump", "peace", "dance"), "none", "toggle", "smile", "L"),
    L(54.86, 56.72, "PLAY", Z, "spin", ("gaming", "dance", "skate"), "game", "none", "smile", "X"),
    L(57.12, 58.60, "正解", Z, "slam", ("run", "jump", "dance"), "none", "door", "smile", "R"),
    L(59.08, 59.55, "NO JUDGE", SPL, "glitch", ("idle",), "none", "versus", "smile", "L"),
    L(61.04, 61.10, "RULES", SPL, "lock", ("idle",), "none", "versus", "smile", "R"),
    L(62.98, 64.60, "STAGE", SPL, "slam", ("idle",), "none", "versus", "smile", "X"),
    L(65.30, 66.64, "非対称戦", SPL, "slam", ("fist",), "none", "versus", "smile", "B"),
    L(67.46, 68.14, "FADE", BUB, "fade", ("celebrate", "disco"), "glass", "crowd", "smile", "L"),
    L(68.68, 70.14, "ROAD", MIL, "slide", ("walk", "present"), "tab", "none", "cool", "R"),
    L(71.48, 72.10, "HOLD", ICE, "lock", ("push", "fist", "reach"), "case", "frost", "grit", "B"),
    L(73.80, 74.40, "SWAY", Z, "wave", ("dance", "jump"), "game", "none", "smile", "L"),
    L(91.62, 91.62, "NOT OVER", ICE, "slam", ("fist", "reach"), "case", "dawnfx", "smile"),
    L(94.06, 94.06, "SIGN", ICE, "fade", ("walk", "push"), "case", "dawnfx", "smile"),
    L(96.26, 96.26, "TIME", ICE, "fill", ("idle", "wipe", "reach"), "case", "clock", "smile"),
    L(98.10, 98.10, "STILL HERE", ICE, "slam", ("fist", "walk"), "case", "dawnfx", "smile"),
]
def act_of(l, t):
    a = l["acts"]; return a[int(max(0, t - l["t"]) / 1.03) % len(a)]
ACTS_GEN = {BUB: ("celebrate", "disco", "toast", "strut"), MIL: ("cool", "present", "type", "watch"), ICE: ("fist", "push", "reach", "run"), Z: ("dance", "jump", "gaming", "skate")}
LINE_END = {i: (LINES[i + 1]["t"] if i + 1 < len(LINES) else 101.0) for i in range(len(LINES))}
LINE_END[27] = 76.3; LINE_END[31] = 101.0

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
    rng_(18.2, 67.46, 2); rng_(67.46, 76.3, 1); rng_(76.3, 91.62, 1); rng_(91.62, 101, 1); rng_(101, DUR, 4)
    for l in LINES: cuts.add(l["t"]); cuts.add(round(l["w0"] - .05, 3))
    for t0 in INTRO_CARDS + [13.4, 15.4, 18.2, 76.3, 91.62, 101.0]: cuts.add(t0)
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
    BUB: dict(skin=(255, 214, 180), top=(255, 205, 70), top2=(255, 60, 160), pants=(250, 250, 255), shoe=(255, 196, 60), hair=(45, 25, 25), scale=1.0),
    MIL: dict(skin=(240, 205, 175), top=(28, 40, 88), top2=(255, 255, 255), pants=(28, 40, 88), shoe=(20, 20, 30), hair=(30, 25, 32), scale=1.02),
    ICE: dict(skin=(226, 192, 166), top=(96, 100, 112), top2=(150, 52, 56), pants=(72, 74, 84), shoe=(62, 46, 42), hair=(38, 32, 40), scale=1.06),
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
    elif act == "strut":
        ph = t * 4.5; p.update(bob=.02 * abs(s(ph)), hx=.03 * s(ph), lean=-.06, lt=(.5 * s(ph), .4 * max(0, s(ph + 1.6))), rt=(-.5 * s(ph), .4 * max(0, -s(ph + 1.6))), la=(-.35 * s(ph), .5), ra=(.35 * s(ph), .5), tilt=-.12)
    elif act == "toast":
        p.update(bob=.01 * s(t * 4), lean=-.06, lt=(.15, 0), rt=(-.15, 0), la=(.4, -.9), ra=(1.4 + .8 * max(0, s(t * 3)), 1.0), tilt=-.15)
    elif act == "disco":
        ph = t * 6.2; p.update(bob=.02 * abs(s(ph)), hx=.03 * s(ph), lt=(.3 * s(ph), .3), rt=(-.3 * s(ph), .3), la=(1.5 - 1.4 * s(ph), .3), ra=(1.5 + 1.4 * s(ph), .3), tilt=.12 * s(ph))
    elif act == "phoneear":
        p.update(bob=.006 * s(t * 3), lean=-.05, lt=(.16, 0), rt=(-.16, 0), la=(.35, -.8), ra=(.9, 2.0), tilt=.12)
    elif act == "present":
        p.update(bob=.006 * s(t * 3), lean=.04, lt=(.14, 0), rt=(-.14, 0), la=(.2, .25), ra=(1.5 + .1 * s(t * 3), .15))
    elif act == "type":
        p.update(bob=.004 * s(t * 3), lt=(.12, 0), rt=(-.12, 0), la=(1.0 + .2 * s(t * 14), 1.3), ra=(.85, 1.55), tilt=.18)
    elif act == "watch":
        p.update(bob=.004 * s(t * 3), lt=(.12, 0), rt=(-.12, 0), la=(.9, 2.1), ra=(.15, .25), tilt=.22)
    elif act == "cross":
        p.update(bob=.004 * s(t * 3), lt=(.12, 0), rt=(-.12, 0), la=(.75, 1.7), ra=(.55, 2.0), tilt=-.05)
    elif act == "bow":
        k = .5 - .5 * math.cos(t * 2.2); p.update(lean=.15 + .85 * k, tilt=.3 * k, lt=(.1, 0), rt=(-.1, 0), la=(.2 + .3 * k, .1), ra=(.1 + .2 * k, .1))
    elif act == "stumble":
        p.update(lean=.35 + .25 * s(t * 9), bob=.02 * abs(s(t * 9)), lt=(.9 * s(t * 9), .3), rt=(-.5, .2), la=(1.8 + 1.0 * s(t * 11), .6), ra=(2.1 - 1.0 * s(t * 11), .6), tilt=.2)
    elif act == "push":
        p.update(lean=.38, bob=.01 * abs(s(t * 7)), lt=(.75, .1), rt=(-.75 + .2 * s(t * 7), .7), la=(1.5, .2), ra=(1.4, .2), tilt=.15)
    elif act == "reach":
        p.update(bob=.02 * abs(s(t * 5)), lean=-.1, lt=(.1, 0), rt=(-.1, 0), la=(.3, .2), ra=(2.85 + .1 * s(t * 6), .1), tilt=-.35)
    elif act == "wipe":
        p.update(lean=.16 + .02 * s(t * 8), tilt=.3, lt=(.15, 0), rt=(-.15, 0), la=(.15, .2), ra=(1.2, 2.5 + .3 * s(t * 8)))
    elif act == "skate":
        p.update(bob=-.06 + .01 * s(t * 6), lean=.2 + .05 * s(t * 3), hx=.03 * s(t * 3), lt=(.5, .9), rt=(-.3, .9), la=(1.3 + .3 * s(t * 3), .2), ra=(-1.0 - .3 * s(t * 3), .2))
    elif act == "jump":
        p.update(bob=.13 * abs(s(t * 5)), lt=(.5, .9), rt=(-.5, .9), la=(2.6, .3), ra=(2.6, .3), tilt=.1 * s(t * 5))
    elif act == "gaming":
        p.update(bob=.01 * s(t * 9), lean=.06, tilt=.12 * s(t * 9), lt=(.14, 0), rt=(-.14, 0), la=(.85, 1.6), ra=(.95, 1.6))
    elif act == "peace":
        p.update(bob=.02 * abs(s(t * 6)), tilt=-.2, lt=(.16 * s(t * 6), .2), rt=(-.16 * s(t * 6), .2), la=(.2, .2), ra=(2.2 + .15 * s(t * 6), .3))
    elif act == "cool":
        p.update(bob=.005 * s(t * 3), lean=-.02, tilt=-.05 + .03 * s(t * 2), lt=(.1, 0), rt=(-.1, 0), la=(.08, .12), ra=(.75, 1.5))
    elif act == "fist":
        p.update(bob=.006 * s(t * 3), lean=-.05, lt=(.2, 0), rt=(-.2, 0), la=(-.15, .2), ra=(2.95, .1), tilt=-.15)
    else:  # idle
        p.update(bob=.006 * s(t * 3), lt=(.16, 0), rt=(-.16, 0), la=(.2 + .04 * s(t * 3), .25), ra=(-.2, .25))
    return p

@lru_cache(maxsize=None)
def _dummy(): return 0

def person(gen, act, t, h, hp="none", mood="smile", alpha=1.0, flip=False):
    """RGBA layer, feet anchored; returns (layer, anchor_x, anchor_y)."""
    sp = SPEC[gen]; S = 2; hh = h * S * sp["scale"]
    lw, lh = int(1.5 * hh), int(1.32 * hh)
    lay = Image.new("RGBA", (lw, lh), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    ax, ay = lw / 2, lh - .06 * hh
    ol = (20, 20, 35)
    def P(x, y): return (ax + x * hh, ay + y * hh)
    def circ(c, r, fill, outline=None, w=0):
        d.ellipse([c[0] - r * hh, c[1] - r * hh, c[0] + r * hh, c[1] + r * hh], fill=fill, outline=outline, width=w)
    def limb(p0, a1, a2, l1, l2, col, wd):
        k = P(p0[0] + l1 * math.sin(a1), p0[1] + l1 * math.cos(a1))
        e = P(p0[0] + l1 * math.sin(a1) + l2 * math.sin(a2), p0[1] + l1 * math.cos(a1) + l2 * math.cos(a2)); s0 = P(*p0)
        for cc, ww_ in ((ol, wd * hh + 6), (col, wd * hh)):
            d.line([s0, k], fill=cc, width=int(ww_)); d.line([k, e], fill=cc, width=int(ww_))
            for q in (s0, k, e): d.ellipse([q[0] - ww_ / 2, q[1] - ww_ / 2, q[0] + ww_ / 2, q[1] + ww_ / 2], fill=cc)
        return e, k
    pz = pose(act, t)
    hip = (pz["hx"], -.47 - pz["bob"]); lean = pz["lean"]
    sh = (hip[0] + math.sin(lean) * .26, hip[1] - math.cos(lean) * .26)
    hd = (sh[0] + math.sin(lean + pz["tilt"]) * .11, sh[1] - math.cos(lean + pz["tilt"]) * .11)
    slim = gen == MIL

    def leg(side, ang):
        a1, kb = ang
        e, k = limb((hip[0] + side * .025, hip[1]), a1, a1 - kb, .235, .225, sp["pants"], .075 if slim else .085)
        if gen == ICE and side == 1: d.rectangle([k[0] - .02 * hh, k[1] - .02 * hh, k[0] + .03 * hh, k[1] + .03 * hh], fill=(120, 110, 100))   # knee patch
        d.ellipse([e[0] - .03 * hh, e[1] - .035 * hh, e[0] + .10 * hh, e[1] + .03 * hh], fill=sp["shoe"], outline=ol, width=2)
        if gen == ICE: d.polygon([(e[0] + .08 * hh, e[1] + .02 * hh), (e[0] + .13 * hh, e[1] + .04 * hh), (e[0] + .07 * hh, e[1] + .045 * hh)], fill=(90, 70, 60), outline=ol)  # flapping sole
        if gen == BUB: d.polygon([(e[0] - .02 * hh, e[1] - .03 * hh), (e[0] + .02 * hh, e[1] - .06 * hh), (e[0] + .04 * hh, e[1] - .03 * hh)], fill=(255, 255, 255))
        if gen == Z: d.rectangle([e[0] - .03 * hh, e[1] + .012 * hh, e[0] + .10 * hh, e[1] + .03 * hh], fill=(190, 255, 60))
    def arm(side, ang):
        a1, eb = ang
        e, k = limb((sh[0] + side * .02, sh[1] + .015), a1, a1 + eb, .165, .155, sp["top"], .062 if slim else .07)
        if gen == ICE: d.rectangle([k[0] - .022 * hh, k[1] - .022 * hh, k[0] + .022 * hh, k[1] + .022 * hh], fill=(130, 100, 70))          # elbow patch
        if slim: d.ellipse([e[0] - .028 * hh, e[1] - .03 * hh, e[0] + .028 * hh, e[1] + .012 * hh], fill=(255, 255, 255))                  # cuff
        circ(e, .033, sp["skin"], ol, 2)
        if gen == BUB: circ((e[0] + .012 * hh, e[1] + .006 * hh), .012, (255, 220, 80), None); circ((e[0] + .012 * hh, e[1] - .004 * hh), .008, (180, 240, 255))  # ring
        if gen == MIL and side == -1: d.ellipse([e[0] - .02 * hh, e[1] - .03 * hh, e[0] + .02 * hh, e[1] - .005 * hh], fill=(200, 210, 230))   # watch
        return e

    leg(-1, pz["lt"]); arm(-1, pz["la"])
    ws, wh = (.16 if gen == BUB else .10 if slim else .12), (.10 if slim else .095)
    # ---------------- torso
    if gen == ICE:                                    # ill-fitting jacket, ragged hem
        hem = [P(hip[0] + wh + .01 - k * (2 * wh + .02) / 8, hip[1] + .03 + (.025 if k % 2 else 0)) for k in range(9)]
        d.polygon([P(sh[0] - ws, sh[1]), P(sh[0] + ws, sh[1])] + hem, fill=sp["top"], outline=ol)
        for k in range(4): d.line([P(sh[0] - .08 + k * .05, sh[1] + .06 + k * .012), P(sh[0] - .03 + k * .05, sh[1] + .11 + k * .012)], fill=(60, 62, 74), width=3)   # wrinkles
        d.polygon([P(sh[0] - .035, sh[1] + .005), P(sh[0] + .0, sh[1] + .06), P(sh[0] - .06, sh[1] + .05)], fill=(225, 225, 220), outline=ol)        # crooked collar
        d.polygon([P(sh[0] + .012, sh[1] + .03), P(sh[0] + .04, sh[1] + .06), P(sh[0] + .05, sh[1] + .15), P(sh[0] + .02, sh[1] + .16), P(sh[0] + .005, sh[1] + .08)], fill=sp["top2"], outline=ol)  # short crooked tie
        d.line([P(sh[0] - .01, hip[1] - .13), P(sh[0] - .012, hip[1] - .07)], fill=(50, 50, 60), width=2)
    elif gen == MIL:                                  # sharp navy suit
        d.polygon([P(sh[0] - ws, sh[1]), P(sh[0] + ws, sh[1]), P(hip[0] + wh, hip[1] + .06), P(hip[0] - wh, hip[1] + .06)], fill=sp["top"], outline=ol)
        d.polygon([P(sh[0] - .03, sh[1]), P(sh[0] + .03, sh[1]), P(sh[0] + .012, hip[1] - .02), P(sh[0] - .012, hip[1] - .02)], fill=(250, 250, 255))
        d.polygon([P(sh[0] - .008, sh[1] + .02), P(sh[0] + .008, sh[1] + .02), P(sh[0] + .014, sh[1] + .16), P(sh[0], sh[1] + .19), P(sh[0] - .014, sh[1] + .16)], fill=(40, 200, 195))
        d.line([P(sh[0] - .03, sh[1]), P(sh[0] - .075, hip[1] - .06)], fill=(60, 80, 130), width=4); d.line([P(sh[0] + .03, sh[1]), P(sh[0] + .075, hip[1] - .06)], fill=(60, 80, 130), width=4)
        d.polygon([P(sh[0] - .06, sh[1] + .07), P(sh[0] - .03, sh[1] + .07), P(sh[0] - .045, sh[1] + .095)], fill=(255, 255, 255))            # pocket square
    else:
        d.polygon([P(sh[0] - ws, sh[1]), P(sh[0] + ws, sh[1]), P(hip[0] + wh, hip[1] + .01), P(hip[0] - wh, hip[1] + .01)], fill=sp["top"], outline=ol)
    if gen == BUB:                                    # gorgeous: gold sequin jacket, fur stole, chains
        for q in range(26):
            g_ = np.random.default_rng(q); px = sh[0] + (g_.random() - .5) * 2 * ws * .8; py = sh[1] + .03 + g_.random() * (hip[1] - sh[1] - .05)
            circ(P(px, py), .006, (255, 255, 255) if q % 2 else (255, 240, 170))
        for sx_ in (-1, 1): d.polygon([P(sh[0] + sx_ * (ws + .05), sh[1] - .01), P(sh[0] + sx_ * .04, sh[1] - .035), P(sh[0] + sx_ * .04, sh[1] + .04), P(sh[0] + sx_ * ws, sh[1] + .06)], fill=sp["top2"], outline=ol)
        for k in range(9): circ(P(sh[0] - .13 + k * .0325, sh[1] + .005 + .012 * math.sin(k * 1.3)), .034, (255, 250, 250), (225, 220, 235), 1)   # fur stole
        for sx_ in (-1, 1):
            for k in range(3): circ(P(sh[0] + sx_ * (.105 - .005 * k), sh[1] + .05 + k * .04), .028, (255, 250, 250), (225, 220, 235), 1)
        d.arc([*P(sh[0] - .07, sh[1] - .01), *P(sh[0] + .07, sh[1] + .16)], 10, 170, fill=(255, 215, 60), width=5)
        d.arc([*P(sh[0] - .05, sh[1] - .01), *P(sh[0] + .05, sh[1] + .11)], 10, 170, fill=(255, 235, 140), width=4)
        circ(P(sh[0], sh[1] + .17), .022, (255, 215, 60), ol, 2); circ(P(sh[0], sh[1] + .17), .012, (255, 60, 130))
    elif gen == ICE:                                  # ragged muffler + snow
        d.rounded_rectangle([*P(sh[0] - .1, sh[1] - .025), *P(sh[0] + .09, sh[1] + .035)], int(.015 * hh), fill=(130, 140, 155), outline=ol, width=2)
        d.polygon([P(sh[0] + .06, sh[1] + .03), P(sh[0] + .1, sh[1] + .03), P(sh[0] + .11, sh[1] + .17), P(sh[0] + .07, sh[1] + .15)], fill=(130, 140, 155), outline=ol)
        for q in (-.09, -.03, .05, .1): circ(P(sh[0] + q, sh[1] - .035), .013, (245, 250, 255))
    elif gen == Z:
        circ((P(sh[0], sh[1] - .02)[0], P(sh[0], sh[1] - .02)[1]), .075, tuple(int(c * .7) for c in sp["top"]), ol, 2)                                  # hood
        d.line([P(sh[0] - .02, sh[1] + .03), P(sh[0] - .02, sh[1] + .13)], fill=sp["top2"], width=5); d.line([P(sh[0] + .02, sh[1] + .03), P(sh[0] + .02, sh[1] + .12)], fill=sp["top2"], width=5)
        d.rounded_rectangle([*P(hip[0] - .1, hip[1] - .09), *P(hip[0] + .1, hip[1] + .0)], 10, fill=sp["top"], outline=ol)
    leg(1, pz["rt"])
    # ---------------- head
    c = P(*hd); r = .088
    if gen == BUB:
        circ((c[0] - .012 * hh, c[1] - .04 * hh), .12, sp["hair"], ol, 2); circ((c[0] + .03 * hh, c[1] - .08 * hh), .05, (95, 60, 40))
    elif gen == Z: circ((c[0] - .01 * hh, c[1] - .02 * hh), .098, sp["hair"], ol, 2)
    elif gen == MIL: pass
    circ(c, r, sp["skin"], ol, 2)
    if gen == MIL:                                    # neat side-parted hair
        d.pieslice([c[0] - r * hh - 4, c[1] - r * hh - 6, c[0] + r * hh + 4, c[1] + r * hh], 190, 350, fill=sp["hair"], outline=ol); d.line([(c[0] + .012 * hh, c[1] - r * hh), (c[0] - .03 * hh, c[1] - .05 * hh)], fill=(100, 90, 100), width=2)
    if gen == ICE:                                    # messy bedhead
        d.pieslice([c[0] - r * hh - 4, c[1] - r * hh - 6, c[0] + r * hh + 4, c[1] + r * hh], 185, 355, fill=sp["hair"])
        for q in range(6): d.polygon([(c[0] - .07 * hh + q * .026 * hh, c[1] - .075 * hh), (c[0] - .06 * hh + q * .026 * hh + (q % 2) * .02 * hh, c[1] - .135 * hh - (q % 3) * .012 * hh), (c[0] - .045 * hh + q * .026 * hh, c[1] - .075 * hh)], fill=sp["hair"])
        for q in range(14): circ((c[0] - .01 * hh + (q % 7) * .012 * hh, c[1] + .055 * hh + (q // 7) * .014 * hh), .003, (90, 80, 85))     # stubble
    ex, ey = c[0] + .03 * hh, c[1] - .003 * hh
    if gen == BUB:
        for xx in (-.03, .06): d.rounded_rectangle([ex + xx * hh - .04 * hh, ey - .025 * hh, ex + xx * hh + .04 * hh, ey + .03 * hh], 10, fill=(255, 140, 190), outline=(255, 215, 60), width=4)
        d.line([ex + .0 * hh, ey - .005 * hh, ex + .02 * hh, ey - .005 * hh], fill=(255, 215, 60), width=4)
        circ((c[0] - .085 * hh, c[1] + .03 * hh), .014, (255, 215, 60)); circ((c[0] - .085 * hh, c[1] + .055 * hh), .008, (180, 240, 255))
    elif gen == MIL:
        for xx in (-.03, .05): circ((ex + xx * hh, ey), .011, (20, 20, 40))
        for xx in (-.03, .05): d.rectangle([ex + xx * hh - .032 * hh, ey - .022 * hh, ex + xx * hh + .032 * hh, ey + .02 * hh], outline=(30, 30, 45), width=3)
        d.line([ex + .02 * hh, ey - .005 * hh, ex + .02 * hh, ey - .005 * hh], fill=(30, 30, 45), width=3)
        d.line([ex - .06 * hh, ey - .036 * hh, ex - .0 * hh, ey - .042 * hh], fill=(50, 40, 50), width=3)
    else:
        for xx in (-.02, .045): circ((ex + xx * hh - .015 * hh, ey), .011, (20, 20, 40))
    if gen == ICE:                                    # heavy bags, angry-determined brows
        for xx in (-.02, .045): d.arc([ex + xx * hh - .04 * hh, ey - .01 * hh, ex + xx * hh + .01 * hh, ey + .04 * hh], 20, 160, fill=(120, 90, 110), width=3)
        d.line([ex - .055 * hh, ey - .05 * hh, ex - .005 * hh, ey - .028 * hh], fill=(30, 25, 35), width=5); d.line([ex + .02 * hh, ey - .028 * hh, ex + .075 * hh, ey - .052 * hh], fill=(30, 25, 35), width=5)
    if gen == Z:
        d.arc([c[0] - .12 * hh, c[1] - .14 * hh, c[0] + .12 * hh, c[1] + .1 * hh], 180, 360, fill=sp["top2"], width=int(.024 * hh))
        d.rounded_rectangle([c[0] - .125 * hh, c[1] - .04 * hh, c[0] - .07 * hh, c[1] + .06 * hh], 9, fill=sp["top2"], outline=ol, width=2)
        d.rounded_rectangle([c[0] - .108 * hh, c[1] - .02 * hh, c[0] - .087 * hh, c[1] + .04 * hh], 5, fill=(30, 30, 60))
    mx, my = c[0] + .012 * hh, c[1] + .05 * hh
    if mood == "smile": d.arc([mx - .035 * hh, my - .025 * hh, mx + .035 * hh, my + .025 * hh], 10, 170, fill=(60, 20, 30), width=3)
    elif mood == "cool": d.arc([mx - .02 * hh, my - .012 * hh, mx + .04 * hh, my + .018 * hh], 20, 150, fill=(60, 20, 30), width=3)
    elif mood == "grit":
        d.rounded_rectangle([mx - .03 * hh, my - .006 * hh, mx + .03 * hh, my + .02 * hh], 4, fill=(245, 240, 235), outline=(60, 20, 30), width=2); d.line([mx, my - .006 * hh, mx, my + .02 * hh], fill=(60, 20, 30), width=2)
    elif mood == "sad": d.arc([mx - .03 * hh, my - .0 * hh, mx + .03 * hh, my + .04 * hh], 190, 350, fill=(60, 20, 30), width=3)
    else: d.line([mx - .025 * hh, my + .01 * hh, mx + .025 * hh, my + .01 * hh], fill=(60, 20, 30), width=3)
    if gen == ICE:                                    # sweat drops
        for q in range(3):
            u = (t * 1.6 + q * .37) % 1; sx_ = c[0] + (.075 + .03 * q) * hh * (1 if q % 2 else -1.2); sy_ = c[1] - .04 * hh + u * .16 * hh
            d.polygon([(sx_, sy_ - .022 * hh), (sx_ - .011 * hh, sy_), (sx_ + .011 * hh, sy_)], fill=(170, 220, 255)); circ((sx_, sy_), .011, (170, 220, 255))
    e = arm(1, pz["ra"])
    hx, hy = e
    if act == "phoneear" and hp == "none": hp = "brick"
    if gen == Z and hp == "game":                     # headphone cable to the handheld
        d.line([(c[0] - .1 * hh, c[1] + .05 * hh), (c[0] - .12 * hh, c[1] + .25 * hh), (hx - .05 * hh, hy - .02 * hh)], fill=(30, 30, 60), width=3)
    if hp == "glass":
        d.polygon([(hx - 12, hy - 50), (hx + 12, hy - 50), (hx, hy - 20)], fill=(255, 240, 170), outline=(255, 215, 60)); d.line([(hx, hy - 20), (hx, hy + 6)], fill=(255, 235, 150), width=4)
        for q in range(3): circ((hx + (q - 1) * 9, hy - 62 - q % 2 * 8), .004, (255, 255, 255))
    elif hp == "trophy":
        d.polygon([(hx - 34, hy - 100), (hx + 34, hy - 100), (hx + 18, hy - 40), (hx - 18, hy - 40)], fill=(255, 210, 60), outline=ol)
        d.rectangle([hx - 6, hy - 40, hx + 6, hy - 12], fill=(255, 210, 60)); d.rectangle([hx - 24, hy - 12, hx + 24, hy + 4], fill=(230, 170, 40))
    elif hp == "case":                                # cheap, taped-up bag
        d.line([(hx, hy), (hx, hy + 22)], fill=ol, width=5)
        d.rounded_rectangle([hx - 42, hy + 22, hx + 42, hy + 88], 6, fill=(105, 80, 60), outline=ol, width=3); d.line([(hx - 42, hy + 40), (hx + 42, hy + 70)], fill=(200, 190, 160), width=6); d.line([(hx + 20, hy + 22), (hx + 20, hy + 60)], fill=(200, 190, 160), width=5)
    elif hp == "brick":
        d.rounded_rectangle([hx - 12, hy - 78, hx + 14, hy + 6], 6, fill=(30, 30, 40), outline=(255, 215, 60), width=4); d.line([(hx - 4, hy - 78), (hx - 8, hy - 108)], fill=(255, 215, 60), width=5)
    elif hp == "phone":
        d.rounded_rectangle([hx - 12, hy - 40, hx + 14, hy + 4], 5, fill=(20, 20, 40), outline=(120, 240, 255), width=3)
    elif hp == "tab":                                 # tablet with chart
        d.rounded_rectangle([hx - 44, hy - 62, hx + 30, hy - 8], 6, fill=(25, 30, 45), outline=(200, 210, 230), width=3)
        d.line([(hx - 34, hy - 20), (hx - 18, hy - 32), (hx - 4, hy - 26), (hx + 20, hy - 50)], fill=(40, 230, 215), width=4)
    elif hp == "game":                                # handheld console (blue / red grips)
        d.rounded_rectangle([hx - 60, hy - 46, hx + 44, hy], 8, fill=(30, 30, 45), outline=ol, width=3)
        d.rounded_rectangle([hx - 60, hy - 46, hx - 38, hy], 8, fill=(60, 200, 255)); d.rounded_rectangle([hx + 22, hy - 46, hx + 44, hy], 8, fill=(255, 70, 90))
        d.rectangle([hx - 34, hy - 40, hx + 18, hy - 8], fill=(190, 255, 120))
        for q in range(3): d.rectangle([hx - 28 + q * 14, hy - 26 + (q % 2) * 6, hx - 20 + q * 14, hy - 18 + (q % 2) * 6], fill=(30, 80, 30))
    elif hp == "book":
        d.rectangle([hx - 34, hy - 44, hx + 34, hy + 10], fill=(235, 225, 200), outline=ol, width=3)
        d.text((hx, hy - 17), "?", font=font("lat", 40), fill=(120, 60, 60), anchor="mm")
    elif hp == "paper":                               # bundle of résumés
        for q in range(4): d.rectangle([hx - 26 + q * 3, hy - 50 - q * 4, hx + 26 + q * 3, hy + 6 - q * 4], fill=(255 - q * 6, 255 - q * 6, 250 - q * 8), outline=ol, width=2)
        for q in range(4): d.line([(hx - 16, hy - 40 + q * 11), (hx + 20, hy - 40 + q * 11)], fill=(150, 150, 170), width=3)
    lay = lay.resize((lw // S, lh // S), Image.LANCZOS)
    if flip: lay = lay.transpose(Image.FLIP_LEFT_RIGHT)
    if alpha < 1: lay.putalpha(lay.getchannel("A").point(lambda v: int(v * alpha)))
    return lay, lay.width / 2, lay.height - .06 * hh / S

def place(im, gen, act, t, h, x, feet, flip=False, hp="none", mood="smile", alpha=1.0):
    d = ImageDraw.Draw(im); w = h * .28
    d.ellipse([x - w, feet - 14, x + w, feet + 14], fill=(0, 0, 0))                       # ground shadow
    lay, ax, ay = person(gen, act, t, h, hp, mood, alpha, flip)
    im.paste(lay, (int(x - ax), int(feet - ay)), lay)
    if gen == BUB:                                                   # glitter
        for k in range(14):
            ph = (t * 1.3 + k * .37) % 1; a_ = k * 2.4; rr = h * (.25 + .35 * ((k * 37) % 10) / 10)
            sx_, sy_ = x + math.cos(a_) * rr, feet - h * .5 + math.sin(a_ * 1.3) * rr * .9; z = math.sin(ph * math.pi) * (10 + 10 * (k % 3))
            d.polygon([(sx_, sy_ - z), (sx_ + z * .25, sy_ - z * .25), (sx_ + z, sy_), (sx_ + z * .25, sy_ + z * .25), (sx_, sy_ + z), (sx_ - z * .25, sy_ + z * .25), (sx_ - z, sy_), (sx_ - z * .25, sy_ - z * .25)], fill=(255, 250, 200))

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
    u = clamp((t - 88) / 14) if u is None else u
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
                place(im, gen, ACTS_GEN[gen][(int(t / 1.03) + gi) % 4], t + gi, 360 if gen != ICE else 420, x, FEET, gen in (ICE, Z))
            return im
        place(im, ICE if i % 2 == 0 else BUB, ("push", "reach", "run")[int(t / 1.03) % 3] if i % 2 == 0 else ("celebrate", "toast", "disco")[int(t / 1.03) % 3], t, hs, 500, FEET, False, "case" if i % 2 == 0 else "glass", "grit" if i % 2 == 0 else "smile")
        place(im, Z if i % 2 == 0 else MIL, ("dance", "jump", "gaming")[int(t / 1.03) % 3] if i % 2 == 0 else ("present", "type", "cool")[int(t / 1.03) % 3], t, hs, 1400, FEET, True, "game" if i % 2 == 0 else "tab", "smile" if i % 2 == 0 else "cool")
        return im
    if w == ICE and l["wp"] == "dawnfx": im, d = bg_dawn(t, v)
    else: im, d = BG[w](t, v)
    gx = gen_pos(i, v)
    wprops(d, l["wp"], t, lt, T, gx - 450 if i % 2 else gx, h, l)
    if l["wp"] == "crowd" and v == 0:
        for k, (dx, hs, ph) in enumerate([(-420, 400, 1.3), (420, 420, 2.2)]):
            place(im, BUB, ("disco", "toast", "strut")[k + int(t / 1.03) % 2], t + ph, hs, gx + dx, FEET - 40, k == 1, "glass" if k else "none")
    if l["w"] == BUB and l["wp"] == "card": al = 1 - .8 * clamp(lt / T)
    else: al = 1.0
    act = act_of(l, t)
    place(im, w, act, t, h, gx, FEET + (30 if v else 0), i % 2 == 1 and act not in ("run", "stumble", "push"), l["hp"], l["mood"], al)
    if l["wp"] == "frost": wprops(ImageDraw.Draw(im), "frost", t, lt, T, gx, h, l)
    return im

def scene_generic(kind, t, v, gen=None, lt=0):
    if kind == "card":
        im, d = bg_card(t, gen, v); a, b = COL[gen]
        act = ACTS_GEN[gen][int(t / 1.03) % 4]
        hp = {BUB: "glass", MIL: "tab", ICE: "case", Z: "game"}[gen]
        e = ease(lt / .35); xoff = (1 - e) * 900 * (1 if gen in (BUB, ICE) else -1)
        place(im, gen, act, t, 640 if v == 0 else 720, 960 + xoff, FEET + 30, gen in (MIL, Z), hp, "grit" if gen == ICE else "cool" if gen == MIL else "smile")
        return im
    if kind == "lineup":
        im, d = bg_ice(t, v); im = Image.blend(im, Image.new("RGB", (WW, WH), (10, 15, 40)), .35)
        for gi, (g_, x, h) in enumerate([(BUB, 640, 470), (MIL, 830, 470), (ICE, 1030, 560), (Z, 1260, 470)]):
            place(im, g_, ACTS_GEN[g_][(int(t / 1.03) + gi) % 4], t + gi, h, x, FEET, g_ in (ICE, Z),
                  {BUB: "glass", MIL: "tab", ICE: "case", Z: "game"}[g_], "grit" if g_ == ICE else "cool" if g_ == MIL else "smile")
        return im
    if kind == "hero":
        im, d = bg_dawn(t, v, .35); place(im, ICE, ("run", "push", "reach", "fist")[int(t / 1.03) % 4] if v == 0 else ("fist", "reach")[int(t / 1.03) % 2], t, 560 if v == 0 else 700, 960, FEET, False, "case", "smile"); return im
    if kind == "glitch":
        im = Image.new("RGB", (WW, WH), (8, 8, 18)); d = ImageDraw.Draw(im)
        for k in range(40): d.line([(0, (k * 97 + t * 900) % WH), (WW, (k * 97 + t * 900) % WH)], fill=(30, 40, 90), width=2)
        g_ = [BUB, MIL, ICE, Z][int(t * 2 / .513) % 4]
        place(im, g_, "idle", t, 640, 960, FEET + 30); return im
    return bg_ice(t, v)[0]

def scene_for(t, shot):
    v = shot % 2; i = line_at(t)
    if i is not None: return scene_line(i, t, v)
    if t < 5.2:
        k = (shot + 2) % 4; return scene_generic("card", t, v, GEN[k][2], (t - CUTS[shot]) + .2)
    if t < 13.4:
        k = max(j for j, s in enumerate(INTRO_CARDS) if s <= t); return scene_generic("card", t, v, GEN[k][2], t - INTRO_CARDS[k])
    if t < 18.2: return scene_generic("lineup", t, v)
    if t < 91.62:
        if t < 81: return scene_generic("lineup", t, v)
        if t < 86.5:
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

def put(ov, ch, fk, size, cx, cy, sx=1.0, sy=1.0, rot=0.0, alpha=1.0, fill=(255, 255, 255), sw=4, sc=(8, 10, 28), shadow=False):
    if alpha <= .01 or sx == 0: return
    tile, _ = glyph(ch, fk, int(size), fill + (255,), sw, sc + (255,))
    if sx != 1 or sy != 1: tile = tile.resize((max(1, int(tile.width * abs(sx))), max(1, int(tile.height * sy))), Image.BILINEAR)
    if rot: tile = tile.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1: tile = tile.copy(); tile.putalpha(tile.getchannel("A").point(lambda v: int(v * alpha)))
    x, y = int(cx - tile.width / 2), int(cy - tile.height / 2)
    if shadow:
        sh = Image.new("RGBA", tile.size, (0, 0, 0, 0)); sh.putalpha(tile.getchannel("A").point(lambda v: int(v * .45))); ov.paste(sh, (x + 8, y + 10), sh)
    ov.paste(tile, (x, y), tile)

def kinetic(ov, word, style, lt, T, cx, cy, size, acc, acc2, t, wt="B"):
    isjp = any(ord(c) > 255 for c in word)
    fk = ({"B": "jp", "R": "jpr", "L": "jpl", "X": "jpt"} if isjp else {"B": "lat", "R": "latR", "L": "latL", "X": "latX"})[wt]
    spf = {"B": .04, "R": .07, "L": .11, "X": .16}[wt]; swb = {"B": 4, "R": 2}.get(wt, 0)
    f = font(fk, size); sp = size * spf
    advs = [f.getlength(c) for c in word]; total = sum(advs) + sp * (len(word) - 1)
    maxw = 2 * min(cx, W - cx) - 40
    if total > maxw: size = size * maxw / total; f = font(fk, size); advs = [f.getlength(c) for c in word]; sp = size * spf; total = sum(advs) + sp * (len(word) - 1)
    d = ImageDraw.Draw(ov); n = len(word); x = cx - total / 2
    outa = 1 - clamp((lt - (T - .12)) / .12)
    ring_t = lt - .18
    centers = []
    for c_, a_ in zip(word, advs): centers.append(x + a_ / 2); x += a_ + sp
    if style == "node":
        for a, b in zip(centers, centers[1:]): d.line([(a, cy), (b, cy)], fill=acc + (int(220 * outa),), width=4)
    if style == "lock":
        e = ease(lt / .3); m = (1 - e) * 120 + 20; L_ = size * .5
        for sx_, sy_ in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            px, py = cx + sx_ * (total / 2 + m), cy + sy_ * (size * .45 + m * .5)
            d.line([(px, py), (px - sx_ * L_ * .6, py)], fill=acc + (int(255 * outa),), width=10); d.line([(px, py), (px, py - sy_ * L_ * .6)], fill=acc + (int(255 * outa),), width=10)
    if style in ("slam", "lock") and 0 < ring_t < .5:
        r = 60 + ring_t * 1800; d.ellipse([cx - r, cy - r * .35, cx + r, cy + r * .35], outline=acc + (int(200 * (1 - ring_t / .5)),), width=3)
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
        lay.putalpha(Image.composite(lay.getchannel("A"), Image.new("L", ov.size, 0), m)); ov.paste(lay, (0, 0), lay); return total, size
    for i_, (c_, cxx) in enumerate(zip(word, centers)):
        li = lt - i_ * (.03 if style not in ("type", "vanish", "node") else .07 if style != "type" else .05)
        e = ease(li / .11); a = e * outa; sx = sy = 1.0; dx = dy = 0.0; rot = 0.0; col = (255, 255, 255); sc = acc; sw = swb
        if style == "slam": sx = sy = 1 + (1 - e) * 1.6; dy = -(1 - e) * 260; rot = (1 - e) * 8 * (1 if i_ % 2 else -1); dx = math.sin(li * 90) * 10 * clamp(1 - (li - .2) * 4) * (li > .2)
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
        elif style == "slide": dx = -(1 - e) * 1300 + lt * 25; sx = 1 + (1 - e) * 1.2; sy = 1 - (1 - e) * .2
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
        d.rectangle([cx - total / 2, cy + size * .55, cx - total / 2 + total * ease(lt / .25), cy + size * .55 + 6], fill=acc + (int(255 * outa),))
    return total, size

TAG_FONT = "tag"
def gen_tag(ov, gen, lt):
    for jp, en, w in GEN:
        if w == gen:
            a = ease(lt / .18); x = 36 - (1 - a) * 260; d = ImageDraw.Draw(ov); col = COL[w][0]
            d.polygon([(x, 28), (x + 300, 28), (x + 276, 92), (x, 92)], fill=(6, 10, 26, 170), outline=col + (255,))
            for k in range(6): d.rectangle([x + 12 + k * 9, 36, x + 16 + k * 9, 44], fill=col + (255 if (k + int(lt * 12)) % 3 else 90,))
            d.text((x + 14, 66), "[ " + en + " ]", font=font("tag", 34), fill=col + (255,), anchor="lm"); d.text((x + 16, 84), "// " + jp, font=font("jpb", 15), fill=(255, 255, 255, 220), anchor="lm")

def hud_box(ov, cx, cy, tw, size, lt, T, acc, label, i, t):
    d = ImageDraw.Draw(ov); e = ease(lt / .15)
    bw = (tw / 2 + 50) * (.6 + .4 * e); bh = (size * .62 + 16) * (.6 + .4 * e); L = 34; A = acc + (int(255 * e),)
    x0, x1, y0, y1 = cx - bw, cx + bw, cy - bh, cy + bh
    for (px, py, sx_, sy_) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(px, py), (px + sx_ * L, py)], fill=A, width=3); d.line([(px, py), (px, py + sy_ * L)], fill=A, width=3)
    fnt = font("tag", 22); g = f"0x{(i * 2654435761 + int(t * 30) * 40503) % 65536:04X}"
    d.text((x0, y0 - 10), "// " + label, font=fnt, fill=A, anchor="ls"); d.text((x1, y0 - 10), g, font=fnt, fill=(255, 255, 255, int(200 * e)), anchor="rs")
    for k in range(0, int(x1 - x0), 16): d.line([(x0 + k, y1 + 10), (x0 + k, y1 + (18 if k % 64 == 0 else 14))], fill=A, width=1)
    mx = x0 + (x1 - x0) * clamp(lt / T); d.rectangle([mx - 4, y1 + 8, mx + 4, y1 + 24], fill=(255, 255, 255, int(255 * e)))
    d.text((x1, y1 + 42), f"T+{lt:04.2f}s", font=fnt, fill=(255, 255, 255, int(200 * e)), anchor="rs")
    if int(t * 6) % 2 == 0: d.rectangle([x0, y1 + 34, x0 + 12, y1 + 46], fill=A)

def fx_layer(ov, burst, t):
    """RGB split + slice glitch + scanlines on a text layer."""
    a = np.array(ov); al = a[..., 3]; rng_ = np.random.default_rng(int(t * FPS) * 7 + 3)
    amt = max(burst, punch(t) * .55 if punch(t) > .55 else 0)
    if amt > .05:
        for _ in range(int(2 + 7 * amt)):
            ys = int(rng_.integers(0, H - 40)); hh = int(rng_.integers(4, 18 + 40 * amt)); dx = int((rng_.random() - .5) * 150 * amt)
            a[ys:ys + hh] = np.roll(a[ys:ys + hh], dx, axis=1)
    al = a[..., 3]; sh = int((1 if t >= 99.14 else 3) + 13 * amt + 3 * af(onset, t))
    def ghost(rgb, dx):
        g = np.zeros_like(a); g[..., 0], g[..., 1], g[..., 2] = rgb; g[..., 3] = (np.roll(al, dx, axis=1) * .8).astype(np.uint8); return Image.fromarray(g)
    main = a.copy(); main[::3, :, 3] = (main[::3, :, 3] * .85).astype(np.uint8)
    out = Image.alpha_composite(ghost((255, 40, 90), -sh), ghost((30, 230, 255), sh)); out = Image.alpha_composite(out, Image.fromarray(main))
    return out

def title_glitch(ov, txt, cx, cy, size, t, alpha=1.0, acc=(175, 225, 255)):
    f = font("jp", size); advs = [f.getlength(c) for c in txt]; total = sum(advs) + size * .04 * (len(txt) - 1); x = cx - total / 2
    for c, a in zip(txt, advs):
        put(ov, c, "jp", size, x + a / 2, cy, alpha=alpha, fill=(255, 255, 255), sw=int(size * .03), sc=(10, 30, 90)); x += a + size * .04

# ---------------------------------------------------------------- overlay
def overlay(t):
    raw, burst = _overlay(t)
    return fx_layer(raw, burst, t)

# ================================================================ finale: word war (87.6 - 101)
WAR0 = 91.62
POOL = "終回収残時代間今夜氷河期声叫戦席順番名"
HITS = [91.62, 92.5, 93.25, 93.8, 94.06, 95.1, 95.82, 96.26, 97.26, 99.14]
HOT = (255, 110, 80); ICEC = (175, 225, 255)
EVENTS = [
    dict(t0=91.62, d=1.6, text="終わってない", st="scream", size=210, cx=640, cy=330, col=ICEC),
    dict(t0=91.95, d=1.0, text="過去の話", st="strike", size=66, cx=250, cy=170, strike=.3, brk=.6),
    dict(t0=92.35, d=1.0, text="自己責任", st="strike", size=66, cx=1030, cy=540, strike=.28, brk=.55),
    dict(t0=93.25, d=1.0, text="言い訳じゃない", st="wipe", size=150, cx=640, cy=350, n=3, col=ICEC),
    dict(t0=94.06, d=1.05, text="終われなかった", st="scramble", size=160, cx=640, cy=300, col=HOT),
    dict(t0=95.1, d=1.2, text="時代のサイン", st="sign", size=128, cx=640, cy=290, col=HOT),
    dict(t0=96.26, d=1.85, text="回収してない", st="tile", size=78, cx=640, cy=360, col=ICEC),
    dict(t0=97.26, d=.9, text="時間", st="clock", size=290, cx=640, cy=345, col=ICEC),
    dict(t0=97.65, d=.45, text="がまだ", st="quiet", size=54, cx=640, cy=575, col=ICEC),
    dict(t0=98.10, d=1.0, text="残ってるだけ", st="quiet", size=70, cx=640, cy=330, col=ICEC),
    dict(t0=99.14, d=1.75, text="それだけだ", st="final", size=200, cx=640, cy=330, col=(255, 255, 255)),
]
def hit_env(t):
    hs = [h for h in HITS if h <= t]
    return math.exp(-(t - hs[-1]) * 7) if hs else 0.0

def _layout(text, fk, size, maxw=1160):
    f = font(fk, size); adv = [f.getlength(c) for c in text]; sp = size * .03; tot = sum(adv) + sp * (len(text) - 1)
    if tot > maxw: size = size * maxw / tot; f = font(fk, size); adv = [f.getlength(c) for c in text]; sp = size * .03; tot = sum(adv) + sp * (len(text) - 1)
    xs = []; x = 0
    for a in adv: xs.append(x + a / 2); x += a + sp
    return size, xs, tot

def paste_img(ov, img, cx, cy, rot=0.0, alpha=1.0):
    if alpha <= .01: return
    if rot: img = img.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1: img = img.copy(); img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
    ov.paste(img, (int(cx - img.width / 2), int(cy - img.height / 2)), img)

def ev_scream(ov, e, lt, t):
    d = ImageDraw.Draw(ov); u = lt / e["d"]; size, xs, tot = _layout(e["text"], "jp", e["size"] * (.86 + .32 * ease(u * 1.1)))
    g = np.random.default_rng(int(t * FPS) + 5)
    if lt < .45:                                                       # radial shout lines
        for k in range(34):
            a = k / 34 * 6.283 + g.random() * .1; r0 = 120 + 500 * ease(lt / .3); r1 = r0 + 80 + 260 * g.random()
            d.line([(e["cx"] + math.cos(a) * r0, e["cy"] + math.sin(a) * r0 * .6), (e["cx"] + math.cos(a) * r1, e["cy"] + math.sin(a) * r1 * .6)], fill=e["col"] + (int(220 * (1 - lt / .45)),), width=4)
    for k, sc_ in ((3, 1.24), (2, 1.14), (1, 1.06)):                   # reverberation echoes
        for c_, x in zip(e["text"], xs):
            put(ov, c_, "jp", size * sc_, e["cx"] - tot / 2 + x + (x - tot / 2) * (sc_ - 1), e["cy"], alpha=.12 + .05 * (4 - k), fill=e["col"], sw=0)
    for c_, x in zip(e["text"], xs):
        put(ov, c_, "jp", size, e["cx"] - tot / 2 + x + (g.random() - .5) * 10, e["cy"] + (g.random() - .5) * 10, fill=(255, 255, 255), sw=7, sc=e["col"])

def ev_strike(ov, e, lt, t):
    d = ImageDraw.Draw(ov); size, xs, tot = _layout(e["text"], "jpl", e["size"], 600); a_in = ease(lt / .12); x0 = e["cx"] - tot / 2
    if lt < e["brk"]:
        for c_, x in zip(e["text"], xs): put(ov, c_, "jpl", size, x0 + x, e["cy"], alpha=a_in, fill=(190, 200, 215), sw=0)
        if lt > e["strike"]:
            k = ease((lt - e["strike"]) / .12); d.line([(x0 - 14, e["cy"] + size * .25), (x0 - 14 + (tot + 28) * k, e["cy"] - size * .25)], fill=(255, 70, 90, 255), width=7)
    else:                                                               # shatter
        bt = lt - e["brk"]
        for i_, (c_, x) in enumerate(zip(e["text"], xs)):
            tile, _ = glyph(c_, "jpl", int(size), (190, 200, 215, 255), 0, (0, 0, 0, 0)); h2 = tile.height // 2
            top, bot = tile.crop((0, 0, tile.width, h2)), tile.crop((0, h2, tile.width, tile.height)); fade = 1 - clamp(bt / .3)
            sgn = -1 if i_ % 2 else 1
            paste_img(ov, top, x0 + x - sgn * 220 * bt, e["cy"] - h2 / 2 - 260 * bt + 900 * bt * bt * .3, -sgn * 300 * bt, fade)
            paste_img(ov, bot, x0 + x + sgn * 260 * bt, e["cy"] + h2 / 2 + 60 * bt + 900 * bt * bt, sgn * 260 * bt, fade)

def ev_wipe(ov, e, lt, t):
    d = ImageDraw.Draw(ov); size, xs, tot = _layout(e["text"], "jp", e["size"]); x0 = e["cx"] - tot / 2; k = ease(lt / .3)
    lay = Image.new("RGBA", ov.size, (0, 0, 0, 0)); struck = lt > .55
    for i_, (c_, x) in enumerate(zip(e["text"], xs)):
        if i_ < e["n"] and struck: put(lay, c_, "jpl", size, x0 + x, e["cy"], alpha=.6, fill=(150, 160, 180), sw=0)
        else: put(lay, c_, "jp", size, x0 + x, e["cy"], fill=(255, 255, 255), sw=6, sc=e["col"])
    m = Image.new("L", ov.size, 0); ImageDraw.Draw(m).rectangle([0, 0, x0 - 30 + (tot + 60) * k, ov.height], fill=255)
    lay.putalpha(Image.composite(lay.getchannel("A"), Image.new("L", ov.size, 0), m)); ov.paste(lay, (0, 0), lay)
    if lt < .32: d.rectangle([x0 - 30 + (tot + 60) * k - 8, e["cy"] - size * .7, x0 - 30 + (tot + 60) * k + 8, e["cy"] + size * .7], fill=e["col"] + (255,))
    if struck:
        w_ = sum(xs[e["n"] - 1] - xs[0] for _ in [0]) + size * 1.1; kk = ease((lt - .55) / .15)
        d.line([(x0 - 10, e["cy"]), (x0 - 10 + (xs[e["n"] - 1] + size * .55) * kk, e["cy"])], fill=(255, 70, 90, 255), width=10)

def ev_scramble(ov, e, lt, t):
    size, xs, tot = _layout(e["text"], "jp", e["size"]); x0 = e["cx"] - tot / 2; g = np.random.default_rng(int(t * FPS) * 3 + 1)
    for i_, (c_, x) in enumerate(zip(e["text"], xs)):
        rt = .12 + i_ * .06; s0 = np.random.default_rng(i_ + 11); ang = s0.random() * 6.283; k = 1 - ease(lt / rt)
        px, py = x0 + x + math.cos(ang) * 620 * k, e["cy"] + math.sin(ang) * 380 * k
        if lt < rt: put(ov, POOL[int(g.integers(0, len(POOL)))], "jp", size, px, py, alpha=.85, fill=(200, 210, 230), sw=3, sc=e["col"])
        else:
            pop = 1 + .5 * (1 - ease((lt - rt) / .15)); put(ov, c_, "jp", size, x0 + x, e["cy"], pop, pop, 0, 1, (255, 255, 255), 6, e["col"])
    if lt > .12 + len(e["text"]) * .06:                                  # accent underline
        ImageDraw.Draw(ov).rectangle([x0, e["cy"] + size * .62, x0 + tot * ease((lt - .55) / .25), e["cy"] + size * .62 + 5], fill=e["col"] + (255,))

def ev_sign(ov, e, lt, t):
    d = ImageDraw.Draw(ov); a_in = ease(lt / .15)
    hb = max(0, math.sin(lt * 9.8)) ** 8; s = 1 + .16 * hb
    f = "jp"; size = e["size"]; head, tail = "時代の", "サイン"
    hs, hx, ht = _layout(head, "jpl", size * .8, 700); ts, tx, tt = _layout(tail, f, size * 1.25 * s, 700)
    total = ht + tt + 40; x0 = e["cx"] - total / 2
    for c_, x in zip(head, hx): put(ov, c_, "jpl", hs, x0 + x, e["cy"] + 20, alpha=a_in, fill=(220, 230, 245), sw=0)
    for c_, x in zip(tail, tx): put(ov, c_, f, ts, x0 + ht + 40 + x, e["cy"], alpha=a_in, fill=(255, 255, 255), sw=6, sc=e["col"])
    pts = []; n = int(clamp(lt / .5) * 60) + 2                           # ECG line
    for k in range(n):
        x = k / 60 * W; ph = (x / W * 4 - lt * 1.5) % 1
        y = 520 - (90 * math.exp(-((ph - .3) / .02) ** 2) - 50 * math.exp(-((ph - .34) / .02) ** 2) + 12 * math.sin(ph * 30) * .2)
        pts.append((x, y))
    d.line(pts, fill=e["col"] + (255,), width=4)

def ev_tile(ov, e, lt, t):
    a = .3 * min(1, lt / .25, (e["d"] - lt) / .25); size = e["size"]; f = font("jpb", size); wtxt = f.getlength(e["text"]) + size
    for r in range(8):
        y = 20 + r * size * 1.0 + 40; off = (lt * (200 if r % 2 else -200) + r * 130) % wtxt
        for k in range(-1, int(W / wtxt) + 2):
            x = k * wtxt - off
            for j, c_ in enumerate(e["text"]): put(ov, c_, "jpb", size, x + f.getlength(e["text"][:j]) + f.getlength(c_) / 2, y, alpha=a, fill=e["col"], sw=0)

def ev_clock(ov, e, lt, t):
    d = ImageDraw.Draw(ov); a_in = ease(lt / .15); cx, cy, r = e["cx"], e["cy"], 250
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(175, 225, 255, 80), width=3)
    d.arc([cx - r, cy - r, cx + r, cy + r], -90, -90 + 360 * (.34 + .02 * math.sin(lt * 8)), fill=(255, 255, 255, 255), width=14)
    for k in range(60): a = k / 60 * 6.283; d.line([(cx + math.cos(a) * (r - 6), cy + math.sin(a) * (r - 6)), (cx + math.cos(a) * (r - (26 if k % 5 == 0 else 14)), cy + math.sin(a) * (r - (26 if k % 5 == 0 else 14)))], fill=(175, 225, 255, 160), width=2)
    size, xs, tot = _layout(e["text"], "jp", e["size"], 460)
    for c_, x in zip(e["text"], xs): put(ov, c_, "jp", size, cx - tot / 2 + x, cy, alpha=a_in, fill=(255, 255, 255), sw=6, sc=e["col"])

def ev_quiet(ov, e, lt, t):
    a = min(1, lt / .3, (e["d"] - lt) / .12); size, xs, tot = _layout(e["text"], "jpl", e["size"], 900); sp = size * .3
    n = len(e["text"]); tw = tot + sp * (n - 1); x = e["cx"] - tw / 2
    for c_, x_ in zip(e["text"], xs): put(ov, c_, "jpl", size, e["cx"] - tot / 2 + x_ + sp * (xs.index(x_) - (n - 1) / 2), e["cy"], alpha=a, fill=(230, 240, 255), sw=0)

def ev_final(ov, e, lt, t):
    FK = "jpl"
    a = min(ease(lt / .25), 1 - clamp((lt - (e["d"] - .2)) / .2)); size, xs, tot = _layout(e["text"], "jpl", e["size"] * (.94 + .06 * ease(lt / 1.0)), 1100)
    lay = Image.new("RGBA", ov.size, (0, 0, 0, 0))
    for c_, x in zip(e["text"], xs): put(lay, c_, FK, size, e["cx"] - tot / 2 + x, e["cy"], fill=(255, 255, 255), sw=0)
    glow = lay.filter(ImageFilter.GaussianBlur(14)); glow = Image.fromarray(np.dstack([np.full(lay.size[::-1], 175, np.uint8), np.full(lay.size[::-1], 225, np.uint8), np.full(lay.size[::-1], 255, np.uint8), np.asarray(glow.getchannel("A"))]))
    out = Image.alpha_composite(glow, lay)
    out.putalpha(out.getchannel("A").point(lambda v: int(v * a))); ov.paste(out, (0, 0), out)

WAR_ST = dict(scream=ev_scream, strike=ev_strike, wipe=ev_wipe, scramble=ev_scramble, sign=ev_sign, tile=ev_tile, clock=ev_clock, quiet=ev_quiet, final=ev_final)
def war_overlay(t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for e in sorted(EVENTS, key=lambda e: e["st"] != "tile"):
        if e["t0"] <= t < e["t0"] + e["d"]: WAR_ST[e["st"]](ov, e, t - e["t0"], t)
    return ov, (0.0 if t >= 99.14 else hit_env(t))

def burst_general(t):
    if 2.1 <= t < 4.8: return clamp(1 - ((t - 2.1) % .513) / .2)
    if 5.2 <= t < 13.4: return clamp(1 - ((t - 5.2) % 2.05) / .25)
    if 15.4 <= t < 15.7 or t >= 101: return clamp(1 - ((t - 15.4) if t < 50 else (t - 101)) / .3)
    return 0.0

def thin_text(ov, txt, cx, cy, size, a, spacing=.3, fk="latl"):
    f = font(fk, size); advs = [f.getlength(c) for c in txt]; sp_ = size * spacing; total = sum(advs) + sp_ * (len(txt) - 1); x = cx - total / 2
    for c, ad in zip(txt, advs):
        put(ov, c, fk, size, x + ad / 2, cy, alpha=a, fill=(235, 245, 255), sw=0); x += ad + sp_
    d = ImageDraw.Draw(ov); d.line([(cx - total / 2, cy + size * .85), (cx + total / 2, cy + size * .85)], fill=(200, 230, 255, int(120 * a)), width=1)

def _overlay(t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); p = punch(t); i = line_at(t); cx = W // 2
    if t < 18.2: return ov, 0.0
    if i is not None and i >= 28: return war_overlay(t)
    if i is not None:
        l = LINES[i]; acc, acc2 = COL[l["w"]]; ws = l["w0"] - .06; WT = 1.2; wl = t - ws
        gen_tag(ov, l["w"] if l["w"] != SPL else ICE, t - l["t"])
        if 0 <= wl < WT:
            pos = (860, 150) if i % 2 == 0 else (420, 150)
            if len(l["big"]) > 5: pos = (cx, 150)
            sz = (230 if len(l["big"]) <= 6 else 165) * (1 if l["wt"] == "B" else .82)
            if l["big"] == "非対称戦": sz = 190
            tot, sz2 = kinetic(ov, l["big"], l["st"], wl, WT, pos[0], pos[1], sz * (1 + .03 * p), acc, acc2, t, l["wt"])
            hud_box(ov, pos[0], pos[1], tot, sz2, wl, WT, acc, {BUB: "BUBBLE", MIL: "MILLENNIAL", ICE: "ICE AGE", Z: "GEN Z", SPL: "ASYMMETRIC"}[l["w"]] + f" / SIG-{i:02d}", i, t)
            return ov, clamp(1 - wl / .25)
        return ov, 0.0
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
                e = ease(li / .08); s = 1 + (1 - e) * 1.6; shake = math.sin(li * 90) * 14 * clamp(1 - li * 5)
                put(ov, c, "jp", size, x + a / 2 + shake, H // 2 - 20 + shake * .5, s, s, (1 - e) * 6, e, (255, 255, 255), 5, (10, 30, 90))
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
    elif t < 91.62:
        for k, (txt, st_) in enumerate((("SAME SOCIETY", 77.6), ("DIFFERENT STAGE", 83.2))):
            u = (t - st_) / 4.2
            if 0 <= u <= 1: thin_text(ov, txt, cx, 170, 44, min(1, u / .15, (1 - u) / .15), .35)
    elif t >= 101.0:
        a = clamp((t - 101) / 1.6)
        d.rectangle([0, H // 2 - 110, W, H // 2 + 150], fill=(4, 8, 28, int(150 * a)))
        thin_text(ov, TITLE, cx, H // 2 - 10, 96, a, .45, "jpt")
        thin_text(ov, "produced by okita", cx, H // 2 + 96, 24, a * .9, .5)
    return ov, burst_general(t)

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
    if WAR0 <= t < 101: arr = arr * (.55 if t < 99.14 else .4)
    arr = arr * vignette() + (100 * hit_env(t) if WAR0 <= t < 101 else 0) + 45 * punch(t) * (1 if 67.46 <= t < 76.3 or on > .8 else .3) + 110 * max(0, 1 - (t - CUTS[s]) / .08)
    if punch(t) > .6 and (line_at(t) is not None or t > 75):
        g = np.random.default_rng(int(t * FPS)); ys = int(g.integers(0, H - 60)); hh = int(g.integers(10, 50)); arr[ys:ys + hh] = np.roll(arr[ys:ys + hh], int((g.random() - .5) * 90 * punch(t)), axis=1)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    im = Image.alpha_composite(im, overlay(t)).convert("RGB")
    a2 = np.asarray(im, np.float32); a2[::3] *= .9; yb = int((t * 220) % (H + 80)) - 40; a2[max(0, yb):max(0, yb + 36)] *= 1.08
    im = Image.fromarray(np.clip(a2, 0, 255).astype(np.uint8))
    for on_t, g_ in ((68.14, BUB), (70.14, MIL), (72.10, ICE), (74.40, Z)):
        fade = max(0, 1 - (t - on_t) / .18) if t >= on_t else 0
        if fade: im = Image.blend(im, Image.new("RGB", (W, H), COL[g_][0]), .45 * fade)
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
