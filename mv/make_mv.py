"""Audio-reactive MV generator for 'asymmetric age'.

usage: python mv/make_mv.py <audio.mp3> <out.mp4> [lyrics.json]
lyrics.json (optional): [{"t": 12.5, "end": 16.0, "text": "..."}, ...]
"""
import json, subprocess, sys, math
import numpy as np, librosa, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1280, 720, 30
audio, out = sys.argv[1], sys.argv[2]
lyrics = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else []
TITLE, SUB = "asymmetric age", "produced by okita"

y, sr = librosa.load(audio, sr=22050, mono=True)
dur = len(y) / sr
n_frames = int(dur * FPS)
hop = sr // FPS
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=64)
mel = np.log1p(mel); mel /= np.percentile(mel, 99) + 1e-9; mel = np.clip(mel, 0, 1.2)
rms = librosa.feature.rms(S=S)[0]; rms = rms / (rms.max() + 1e-9)
bass = mel[:8].mean(0)
onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr, hop_length=hop)
onset = onset / (onset.max() + 1e-9)

def smooth(a, k=0.5):
    o = np.zeros_like(a); v = 0
    for i, x in enumerate(a):
        v = max(x, v * k + x * (1 - k)); o[i] = v
    return o
mel_s = np.stack([smooth(m, 0.7) for m in mel])
bass_s, rms_s = smooth(bass, 0.8), smooth(rms, 0.8)

def font(sz, jp=False):
    for p in (["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"] if jp else []) + \
             ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Light.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
f_title, f_sub, f_lyr = font(64), font(26), font(38, True)

rng = np.random.default_rng(7)
N = 160
snow = np.c_[rng.random(N) * W, rng.random(N) * H, rng.random(N) * 2.5 + 0.8, rng.random(N) * 6.28]

gy, gx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
gx /= W // 4; gy /= H // 4
base = np.zeros((H // 4, W // 4, 3), np.float32)
base[..., 0] = 5 + 12 * gy; base[..., 1] = 10 + 28 * gy; base[..., 2] = 26 + 50 * gy

def bg(t, i):
    a = base.copy()
    e = 0.35 + 0.9 * rms_s[i]
    for k, (col, ph, sp) in enumerate([((30, 255, 190), 0.0, 0.11), ((90, 160, 255), 2.1, 0.07), ((190, 110, 255), 4.0, 0.05)]):
        # asymmetric: bands tilt and thicken toward the right
        c = 0.30 + 0.12 * k + 0.10 * np.sin(gx * (3 + k) + t * sp * 6 + ph) + 0.05 * gx
        band = np.exp(-((gy - c) ** 2) / (0.010 + 0.006 * gx)) * (0.5 + 0.5 * np.sin(gx * 9 + t * 0.6 + ph)) ** 2
        for ch in range(3): a[..., ch] += band * col[ch] * e * 0.55
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return im.filter(ImageFilter.GaussianBlur(3))

CX, CY = int(W * 0.36), int(H * 0.54)   # off-centre on purpose
def render(i):
    t = i / FPS
    im = bg(t, i).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    # off-centre moon
    r0 = 92 + 26 * bass_s[i]
    for k in range(6, 0, -1):
        d.ellipse([W * 0.78 - r0 - k * 14, H * 0.2 - r0 - k * 14, W * 0.78 + r0 + k * 14, H * 0.2 + r0 + k * 14],
                  fill=(170, 210, 255, 6))
    d.ellipse([W * 0.78 - r0, H * 0.2 - r0, W * 0.78 + r0, H * 0.2 + r0], fill=(215, 235, 255, 200))
    # asymmetric radial spectrum: long on one side, short on the other
    nb = 64; R = 120 + 30 * bass_s[i]
    for b in range(nb * 2):
        ang = b / (nb * 2) * 2 * math.pi + t * 0.15
        v = mel_s[b % nb if b < nb else nb - 1 - (b % nb), i]
        skew = 1.0 + 0.9 * math.cos(ang - 0.6)
        L = 8 + v * 150 * skew
        c = (int(120 + 100 * v), int(200 + 50 * v), 255, int(140 + 100 * min(v, 1)))
        d.line([CX + R * math.cos(ang), CY + R * math.sin(ang), CX + (R + L) * math.cos(ang), CY + (R + L) * math.sin(ang)], fill=c, width=4)
    d.ellipse([CX - R + 6, CY - R + 6, CX + R - 6, CY + R - 6], outline=(200, 235, 255, 130), width=2)
    # pulse ring on onsets
    pr = R + 6 + 200 * onset[i]
    d.ellipse([CX - pr, CY - pr, CX + pr, CY + pr], outline=(255, 255, 255, int(120 * onset[i])), width=2)
    # snow
    snow[:, 1] += snow[:, 2] * (1 + 2 * rms_s[i]); snow[:, 0] += np.sin(snow[:, 3] + t) * 0.8 + 0.6
    snow[:, 0] %= W; snow[:, 1] %= H
    for x, yy, s, _ in snow:
        d.ellipse([x - s, yy - s, x + s, yy + s], fill=(235, 245, 255, int(80 + 30 * s)))
    # ground reflection line
    d.line([0, H * 0.9, W, H * 0.9], fill=(150, 200, 255, 60), width=1)
    im = Image.alpha_composite(im, ov)
    d = ImageDraw.Draw(im)
    # title (intro / outro)
    a = 1.0 if t < 1 else max(0, 1 - (t - 1) / 3.0)
    a = max(a, min(1, (t - (dur - 5)) / 2)) if t > dur - 5 else a
    if a > 0:
        al = int(255 * a)
        d.text((70, H - 150), TITLE, font=f_title, fill=(235, 246, 255, al))
        d.text((74, H - 76), SUB, font=f_sub, fill=(160, 200, 240, al))
    # lyrics
    for L in lyrics:
        if L["t"] <= t <= L["end"]:
            fa = min(1, (t - L["t"]) / 0.3, (L["end"] - t) / 0.3)
            tw = d.textlength(L["text"], font=f_lyr)
            d.text(((W - tw) / 2, H * 0.83), L["text"], font=f_lyr, fill=(255, 255, 255, int(255 * fa)),
                   stroke_width=2, stroke_fill=(10, 20, 50, int(200 * fa)))
    return im.convert("RGB").tobytes()

ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-i", audio, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-shortest", out], stdin=subprocess.PIPE)
for i in range(n_frames):
    ff.stdin.write(render(min(i, mel_s.shape[1] - 1)))
    if i % 300 == 0: print(i, "/", n_frames, flush=True)
ff.stdin.close(); ff.wait(); print("done", out)
