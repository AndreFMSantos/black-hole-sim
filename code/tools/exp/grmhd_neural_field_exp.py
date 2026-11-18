# Experiment (v2.2f-a2): neural field (hash grid + MLP) vs a regular grid in log10 rho. Usage: uv run --with torch --with h5py --with numpy --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match python -I tools/exp/grmhd_neural_field_exp.py <file.h5>   (uv installed CUDA 13 wheels: it worked on the RTX 5080)
# Neural field (multi-resolution hash grid + MLP) fitted to the log10 rho of the real snapshot, vs a regular grid with the same budget
import sys, time, h5py, numpy as np, torch
dev = 'cuda'
f = h5py.File(sys.argv[1], 'r'); lrho = np.log10(f['prims'][..., 0].astype(np.float64))
lo, hi = lrho.min(), lrho.max()
V = torch.tensor((lrho - lo) / (hi - lo), dtype=torch.float32, device=dev)        # (288,128,128) em [0,1]
N = torch.tensor(V.shape, device=dev)
print("volume", tuple(V.shape), "range", round(hi - lo, 2), "dex")

class HashGrid(torch.nn.Module):
    def __init__(self, levels, log2T, F, base, top):
        super().__init__()
        self.L, self.T, self.F = levels, 1 << log2T, F
        self.res = [int(round(base * (top / base) ** (l / max(1, levels - 1)))) for l in range(levels)]
        self.tab = torch.nn.Parameter(torch.empty(levels, self.T, F).uniform_(-1e-4, 1e-4))
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)
    def forward(self, x):                                   # x: (B,3) em [0,1]
        out = []
        for l in range(self.L):
            r = self.res[l]; p = x * (r - 1); i0 = p.floor().long(); w = p - i0
            acc = 0
            for c in range(8):
                o = torch.tensor([(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1], device=x.device)
                ic = (i0 + o).clamp(max=r - 1)
                h = ((ic[:, 0] * self.primes[0].to(x.device)) ^ (ic[:, 1] * self.primes[1].to(x.device)) ^ (ic[:, 2] * self.primes[2].to(x.device))) % self.T
                wt = torch.prod(torch.where(o.bool(), w, 1 - w), dim=1, keepdim=True)
                acc = acc + wt * self.tab[l][h]
            out.append(acc)
        return torch.cat(out, dim=1)

def run(levels, log2T, F, steps, hid=64, label=''):
    torch.manual_seed(0)
    enc = HashGrid(levels, log2T, F, 8, 288).to(dev)
    mlp = torch.nn.Sequential(torch.nn.Linear(levels * F, hid), torch.nn.ReLU(), torch.nn.Linear(hid, hid), torch.nn.ReLU(), torch.nn.Linear(hid, 1)).to(dev)
    nparam = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in mlp.parameters())
    opt = torch.optim.Adam([{'params': enc.parameters(), 'lr': 5e-2}, {'params': mlp.parameters(), 'lr': 5e-3}])
    t0 = time.time()
    for s in range(steps):
        idx = torch.stack([torch.randint(0, int(N[k]), (1 << 17,), device=dev) for k in range(3)], 1)
        x = (idx.float() + 0.5) / N.float()
        y = V[idx[:, 0], idx[:, 1], idx[:, 2]][:, None]
        loss = torch.nn.functional.mse_loss(mlp(enc(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s == int(steps * 0.7): 
            for g in opt.param_groups: g['lr'] *= 0.2
    torch.cuda.synchronize(); tt = time.time() - t0
    # evaluation on all cells
    err = []
    with torch.no_grad():
        ii = torch.stack(torch.meshgrid([torch.arange(int(N[k]), device=dev) for k in range(3)], indexing='ij'), -1).reshape(-1, 3)
        for a in range(0, ii.shape[0], 1 << 20):
            idx = ii[a:a + (1 << 20)]; x = (idx.float() + 0.5) / N.float()
            err.append((mlp(enc(x))[:, 0] - V[idx[:, 0], idx[:, 1], idx[:, 2]]) * (hi - lo))
    e = torch.cat(err).double().cpu().numpy()
    mb = nparam * 2 / 1e6
    print(f"{label:34s} params {nparam/1e3:7.0f}k ({mb:5.2f} MB fp16)  training {tt:4.0f} s  rms error {np.sqrt((e**2).mean()):.4f} dex  p99.9 {np.percentile(np.abs(e),99.9):.3f} dex", flush=True)

run(8, 15, 2, 2500, label='hash-grid 8 levels, T=2^15')
run(12, 17, 2, 2500, label='hash-grid 12 levels, T=2^17')
