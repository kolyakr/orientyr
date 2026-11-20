# Throwaway feasibility micro-benchmark (not project code): cost of exhaustive
# (x, y, theta, scale) correlation of frame features vs pre-rotated map feature stacks on MPS.
import time, torch
import torch.nn.functional as Fn
dev = "mps"
torch.manual_seed(0)

def bench(fn, n=5):
    fn(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / n * 1000

R, S, C = 36, 3, 64  # 36 headings (10 deg), 3 scales, PCA-64 channels
for (H, W, K, name) in [(131, 260, 14, "stride14 (DINO on full map)"), (231, 456, 25, "stride8 (2x upsampled)")]:
    for dt in (torch.float32, torch.float16):
        m = torch.randn(R, C, H, W, device=dev, dtype=dt)
        k = torch.randn(S, C, K, K, device=dev, dtype=dt)
        try:
            ms = bench(lambda: Fn.conv2d(m, k))
            gf = 2 * R * S * C * K * K * (H - K + 1) * (W - K + 1) / 1e9
            print(f"{name} {str(dt)[6:]} direct conv: {ms:.1f} ms  ({gf:.0f} GFLOP, {gf/ms:.2f} TFLOP/s)")
        except Exception as e:
            print(name, dt, "conv FAIL", repr(e)[:200])
        del m, k
    Hp, Wp = H + K, W + K
    try:
        m = torch.randn(R, C, H, W, device=dev)
        Mf = torch.fft.rfft2(m, s=(Hp, Wp))  # would be precomputed in the Map Pack
        Fq = Mf.shape[-2] * Mf.shape[-1]
        Mr = Mf.real.permute(2, 3, 0, 1).reshape(Fq, R, C).contiguous()
        Mi = Mf.imag.permute(2, 3, 0, 1).reshape(Fq, R, C).contiguous()
        del Mf, m
        k = torch.randn(S, C, K, K, device=dev)
        def query():
            Kf = torch.fft.rfft2(k, s=(Hp, Wp)).conj()  # correlation = conj in freq domain
            Kr = Kf.real.permute(2, 3, 1, 0).reshape(Fq, C, S)
            Ki = Kf.imag.permute(2, 3, 1, 0).reshape(Fq, C, S)
            re = torch.bmm(Mr, Kr) - torch.bmm(Mi, Ki)
            im = torch.bmm(Mr, Ki) + torch.bmm(Mi, Kr)
            P = torch.complex(re, im).reshape(Hp, Wp // 2 + 1, R, S).permute(2, 3, 0, 1)
            vol = torch.fft.irfft2(P, s=(Hp, Wp))
            return torch.softmax(vol.flatten() / 0.1, 0)  # posterior normalisation over x,y,theta,s
        ms = bench(query)
        print(f"{name} FFT + per-frequency bmm + softmax: {ms:.1f} ms  (map FFT stack {2*Mr.numel()*4/1e6:.0f} MB)")
        del Mr, Mi
    except Exception as e:
        print(name, "FFT FAIL", repr(e)[:300])
    torch.mps.empty_cache()

# Histogram Bayes filter step on the full (theta, y, x) grid: predict = blur, update = multiply.
bel = torch.rand(1, 1, R, 131, 260, device=dev); lik = torch.rand_like(bel)
ker = torch.ones(1, 1, 3, 5, 5, device=dev) / 75
def bayes():
    b = Fn.conv3d(Fn.pad(bel, (2, 2, 2, 2, 1, 1), mode="circular"), ker)
    b = b * lik
    return b / b.sum()
print(f"histogram-filter predict+update (36x131x260): {bench(bayes):.1f} ms")

try:
    import timm
    for arch, sz in (("vit_large_patch16_224", 448), ("vit_base_patch16_224", 448)):
        net = timm.create_model(arch, pretrained=False, img_size=sz, num_classes=0).to(dev).eval().half()
        x = torch.randn(3, 3, sz, sz, device=dev, dtype=torch.float16)  # 3 scales batched
        with torch.no_grad():
            ms = bench(lambda: net.forward_features(x), n=3)
        print(f"{arch} fp16 random-init @ {sz}px, batch 3: {ms:.0f} ms")
        del net
except Exception as e:
    print("timm skip:", repr(e)[:200])
