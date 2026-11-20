# Throwaway MPS capability/throughput probe (research only, not project code).
import time, torch, torch.nn as nn, torch.nn.functional as F
dev = "mps"
print("torch", torch.__version__)
print("autocast available on mps:", torch.amp.is_autocast_available("mps"))
lin = nn.Linear(64, 64).to(dev); x = torch.randn(8, 64, device=dev)
for dt in (torch.float16, torch.bfloat16):
    try:
        with torch.autocast("mps", dtype=dt):
            y = lin(x)
        print("autocast", dt, "->", y.dtype)
    except Exception as e:
        print("autocast", dt, "FAIL", repr(e)[:150])
def tryop(name, fn):
    try:
        fn(); torch.mps.synchronize(); print("op OK  ", name)
    except Exception as e:
        print("op FAIL", name, repr(e)[:160])
a = torch.randn(64, 64, device=dev)
tryop("bf16 matmul", lambda: (a.bfloat16() @ a.bfloat16()).sum().item())
tryop("linalg.solve 8x8 batch", lambda: torch.linalg.solve(torch.randn(100, 8, 8, device=dev), torch.randn(100, 8, 1, device=dev)))
tryop("linalg.svd 9x9 batch", lambda: torch.linalg.svd(torch.randn(100, 9, 9, device=dev)))
tryop("fft.rfft2 1024", lambda: torch.fft.rfft2(torch.randn(1, 1024, 1024, device=dev)))
def gs():
    img = torch.randn(4, 3, 256, 256, device=dev, requires_grad=True)
    grid = (torch.rand(4, 128, 128, 2, device=dev) * 2 - 1).requires_grad_()
    F.grid_sample(img, grid, align_corners=False).sum().backward()
tryop("grid_sample fwd+bwd", gs)
tryop("interpolate antialias", lambda: F.interpolate(torch.randn(1, 3, 640, 640, device=dev), size=(200, 200), mode="bilinear", antialias=True))
tryop("topk/cdist", lambda: torch.cdist(torch.randn(2048, 256, device=dev), torch.randn(4096, 256, device=dev)).topk(2, dim=1))

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        B, N, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        x = x + s.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + s.mlp(s.n2(x))
class ViT(nn.Module):  # same shapes/FLOPs as DINOv2 ViT-*/14 (random weights)
    def __init__(s, d, h, depth, p=14):
        super().__init__(); s.pe = nn.Conv2d(3, d, p, p); s.blocks = nn.Sequential(*[Block(d, h) for _ in range(depth)])
    def forward(s, x):
        return s.blocks(s.pe(x).flatten(2).transpose(1, 2))
def timeit(fn, n=5, warm=2):
    for _ in range(warm): fn()
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / n
cfgs = {"ViT-S/14": (384, 6, 12), "ViT-B/14": (768, 12, 12), "ViT-L/14": (1024, 16, 24)}
for name, (d, h, depth) in cfgs.items():
    m = ViT(d, h, depth).to(dev).eval()
    for res, bs in ((224, 32), (448, 8)):
        if name == "ViT-L/14" and res == 448: bs = 4
        xx = torch.randn(bs, 3, res, res, device=dev)
        for mode in ("fp32", "fp16"):
            def f():
                with torch.no_grad():
                    if mode == "fp16":
                        with torch.autocast("mps", dtype=torch.float16): m(xx)
                    else: m(xx)
            try:
                dt = timeit(f)
                print(f"{name} {res}px bs{bs} {mode}: {bs/dt:7.1f} img/s")
            except Exception as e:
                print(f"{name} {res}px {mode} FAIL {repr(e)[:120]}")
    del m; torch.mps.empty_cache()

# ACE-style head: 8x (1x1 conv == Linear) width 512 with skips, trained on buffered features.
class Head(nn.Module):
    def __init__(s, din=384, w=512, out=3):
        super().__init__()
        s.inp = nn.Linear(din, w)
        s.l = nn.ModuleList([nn.Linear(w, w) for _ in range(8)])
        s.out = nn.Linear(w, out)
    def forward(s, x):
        x = F.relu(s.inp(x)); res = x
        for i, l in enumerate(s.l):
            x = F.relu(l(x))
            if i in (2, 5): x = x + res; res = x
        return s.out(x)
for din, w in ((384, 512), (768, 768)):
    head = Head(din, w).to(dev); opt = torch.optim.AdamW(head.parameters(), 1e-3)
    feats = torch.randn(5120, din, device=dev); tgt = torch.randn(5120, 3, device=dev)
    for mode in ("fp32", "fp16"):
        def step():
            opt.zero_grad(set_to_none=True)
            if mode == "fp16":
                with torch.autocast("mps", dtype=torch.float16):
                    loss = F.smooth_l1_loss(head(feats).float(), tgt)
            else:
                loss = F.smooth_l1_loss(head(feats), tgt)
            loss.backward(); opt.step()
        try:
            dt = timeit(step, n=20, warm=3)
            fps = 5120 / dt
            print(f"head din{din} w{w} {mode}: {1/dt:6.1f} it/s (bs5120) = {fps/1e6:.2f} M feat/s; 16 epochs x 8M feats = {16*8e6/fps/60:.1f} min")
        except Exception as e:
            print(f"head {mode} FAIL {repr(e)[:120]}")
print("recommended max working set (GB):", torch.mps.recommended_max_memory() / 1e9 if hasattr(torch.mps, "recommended_max_memory") else "n/a")
