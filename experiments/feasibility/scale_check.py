"""Quick exploratory check: scale ratio crop->map via SIFT+homography (scratch only)."""
import sys, cv2, numpy as np
mp, *crops = sys.argv[1:]
M = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create(nfeatures=60000)
kM, dM = sift.detectAndCompute(M, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
for cp in crops:
    C = cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    best = None
    for f in (0.15, 0.2, 0.25, 0.3, 0.4):
        Cs = cv2.resize(C, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        kC, dC = sift.detectAndCompute(Cs, None)
        if dC is None or len(kC) < 10: continue
        m = flann.knnMatch(dC, dM, k=2)
        g = [a for a, b in m if a.distance < 0.8 * b.distance]
        if len(g) < 8: continue
        src = np.float32([kC[a.queryIdx].pt for a in g]) / f
        dst = np.float32([kM[a.trainIdx].pt for a in g])
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
        if H is None: continue
        n = int(inl.sum())
        if best is None or n > best[0]: best = (n, f, H)
    if best:
        n, f, H = best
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        c = H @ np.array([320, 320, 1.0]); c = c[:2] / c[2]
        print(f"{cp.split('/')[-1]}: inliers={n} at f={f}, map px per crop px={s:.3f} -> crop is {1/s:.2f}x finer; centre=({c[0]:.0f},{c[1]:.0f})")
    else:
        print(cp, "no fit")
