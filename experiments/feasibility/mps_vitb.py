# Re-measure ViT-B/14-shaped forward on MPS (random weights) with more iterations to reduce noise.
import time, torch, torch.nn as nn, torch.nn.functional as F
dev = "mps"
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
m = nn.Sequential(nn.Conv2d(3, 768, 14, 14)).to(dev)
blocks = nn.Sequential(*[Block(768, 12) for _ in range(12)]).to(dev).eval()
def run(x, half):
    with torch.no_grad():
        ctx = torch.autocast("mps", dtype=torch.float16) if half else torch.autocast("mps", enabled=False)
        with ctx:
            blocks(m(x).flatten(2).transpose(1, 2))
for half in (False, True, False):
    x = torch.randn(32, 3, 224, 224, device=dev)
    for _ in range(3): run(x, half)
    torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(10): run(x, half)
    torch.mps.synchronize(); dt = (time.perf_counter() - t) / 10
    print(f"ViT-B/14 224 bs32 {'fp16' if half else 'fp32'}: {32/dt:.1f} img/s")
