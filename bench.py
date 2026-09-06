import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import time, torch, torch.nn as nn
torch.manual_seed(0)
N, D, BATCH, EPOCHS = 253_680, 21, 1024, 5
X = torch.randn(N, D)
y = (torch.rand(N) < 0.14).float()
def make_model():
    return nn.Sequential(nn.Linear(D, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(32, 1))
def run_epoch(model, opt, crit):
    perm = torch.randperm(N)
    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        crit(model(X[idx]).squeeze(1), y[idx]).backward()
        opt.step()
print("logical cores:", os.cpu_count(), flush=True)
for threads in [4, 6, 8, 10, 14]:
    torch.set_num_threads(threads)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.0))
    run_epoch(model, opt, crit)
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        run_epoch(model, opt, crit)
    print(f"threads={threads:2d}  {(time.perf_counter()-t0)/EPOCHS:.3f} sec/epoch", flush=True)