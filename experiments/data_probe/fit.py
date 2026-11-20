"""Fit an analytic generator to the recovered poses: x linear in t, y sinusoidal, heading = path tangent."""
import json, sys
import numpy as np
from scipy.optimize import least_squares
d = json.load(open(sys.argv[1]))
r = [x for x in d["rows"] if x["ok"]]
t = np.array([x["i"] for x in r], float); X = np.array([x["cx"] for x in r]); Y = np.array([x["cy"] for x in r])
A = np.array([x["angle"] for x in r])
def model(p, t):
    a, b, c, amp, P, ph = p
    return a + b * t, c + amp * np.sin(2 * np.pi * t / P + ph)
def res(p):
    x, y = model(p, t); return np.r_[x - X, y - Y]
p0 = [320, 63.4, 988, -493, 50, 0]
fit = least_squares(res, p0)
x, y = model(fit.x, t)
e = np.hypot(x - X, y - Y)
print("params a,b,c,amp,P,phase:", np.round(fit.x, 3))
print(f"position residual vs analytic path: RMS {np.sqrt((e**2).mean()):.3f} px, max {e.max():.3f} px")
# heading from forward finite difference of the measured track vs measured image rotation
hd = np.degrees(np.arctan2(np.diff(Y), np.diff(X)))
diff = (A[:-1] - 90 - hd + 180) % 360 - 180
print(f"image-up vs direction-to-next-frame: mean {diff.mean():.2f} deg, max |dev| {np.abs(diff).max():.2f} deg")
print("mean RGB crop vs map patch:", np.round(np.mean([x['rgb_crop'] for x in r], 0), 1), np.round(np.mean([x['rgb_map'] for x in r], 0), 1))
