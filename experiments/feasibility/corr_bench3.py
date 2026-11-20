# Throwaway: warning emitted by padded rfft2 on MPS, and ViT-L/B encoder cost (random init, plain nn.Transformer).
import time, warnings, torch
import torch.nn as nn
dev = "mps"
def bench(fn, n=3):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / n * 1000
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    X = torch.fft.rfft2(torch.randn(8, 64, 131, 260, device=dev), s=(145, 274)); torch.mps.synchronize()
    print("padded rfft2 device:", X.device, "| warnings:", [str(x.message)[:160] for x in w])
for name, L, D, H in (("ViT-L/16", 24, 1024, 16), ("ViT-B/16", 12, 768, 12), ("ViT-S/16", 12, 384, 6)):
    layer = nn.TransformerEncoderLayer(D, H, 4 * D, batch_first=True, norm_first=True, activation="gelu")
    net = nn.TransformerEncoder(layer, L, enable_nested_tensor=False).to(dev).eval().half()
    for tokens, label in ((784, "448px"), (196, "224px")):
        x = torch.randn(3, tokens + 1, D, device=dev, dtype=torch.float16)
        with torch.no_grad():
            ms = bench(lambda: net(x))
        print(f"{name} fp16 transformer trunk, batch 3 @ {label}: {ms:.0f} ms")
    del net
