# Throwaway: (1) does rfft2 really run on MPS, (2) local tracking-window cost,
# (3) cheaper histogram filter, (4) ViT frame-encoder cost (random init).
import time, warnings, torch
import torch.nn.functional as Fn
warnings.simplefilter("always")
dev = "mps"
def bench(fn, n=5):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / n * 1000
with warnings.catch_warnings(record=True) as w:
    x = torch.randn(64, 131, 260, device=dev); X = torch.fft.rfft2(x); print("rfft2 out device:", X.device)
    for wi in w: print("WARN:", str(wi.message)[:300])
# tracking: 9 headings (+-20 deg around prior) x 3 scales, +-40 cells window
m = torch.randn(9, 64, 80 + 14, 80 + 14, device=dev, dtype=torch.float16)
k = torch.randn(3, 64, 14, 14, device=dev, dtype=torch.float16)
print(f"tracking window conv (9 headings x 3 scales, 81x81 positions): {bench(lambda: Fn.conv2d(m, k)):.1f} ms")
# histogram filter: separable 2D blur per heading + 3-tap heading blur via roll
bel = torch.rand(36, 1, 131, 260, device=dev); lik = torch.rand_like(bel)
g = torch.tensor([1., 4., 6., 4., 1.], device=dev) / 16
def bayes():
    b = Fn.conv2d(Fn.pad(bel, (2, 2, 0, 0), mode="replicate"), g.view(1, 1, 1, 5))
    b = Fn.conv2d(Fn.pad(b, (0, 0, 2, 2), mode="replicate"), g.view(1, 1, 5, 1))
    b = 0.25 * torch.roll(b, 1, 0) + 0.5 * b + 0.25 * torch.roll(b, -1, 0)
    b = b * lik
    return b / b.sum()
print(f"histogram filter separable (36x131x260): {bench(bayes):.1f} ms")
try:
    import torchvision
    from torchvision.models.vision_transformer import VisionTransformer
    for name, L, D, Hh, M in (("ViT-L/16", 24, 1024, 16, 4096), ("ViT-B/16", 12, 768, 12, 3072)):
        net = VisionTransformer(448, 16, L, Hh, D, M).to(dev).eval().half()
        x = torch.randn(3, 3, 448, 448, device=dev, dtype=torch.float16)
        with torch.no_grad():
            ms = bench(lambda: net(x), n=3)
        print(f"{name} fp16 random-init, 448px, batch 3 (3 scales): {ms:.0f} ms")
        del net
except Exception as e:
    print("torchvision skip:", repr(e)[:200])
