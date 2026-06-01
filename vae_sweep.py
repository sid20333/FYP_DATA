
import os, time, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import geometry_check as geom

LATENT_DIMS    = [6, 8, 10, 12]    
EPOCHS         = 2000    
BATCH          = 4096
LR             = 1e-3 
DATA_PATH      = 'training_X_valid_500k.npy'

ONE_DIM = os.environ.get('AE_LATENT_DIM')
if ONE_DIM is not None:
    LATENT_DIMS = [int(ONE_DIM)]
WALL_BUDGET_S = int(os.environ.get('WALL_BUDGET_S', 24 * 3600)) 

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# ── load + normalise to [0,1]
X = np.load(DATA_PATH).astype(np.float32)
LO = geom.BOUNDS[:, 0].astype(np.float32)
HI = geom.BOUNDS[:, 1].astype(np.float32)
Xn = (X - LO) / (HI - LO)
# 90/10 train/val split
rng = np.random.default_rng(0)
idx = rng.permutation(len(Xn))
n_tr = int(0.9 * len(Xn))
Xtr, Xval = Xn[idx[:n_tr]], Xn[idx[n_tr:]]
print(f'train {len(Xtr)}, val {len(Xval)}, input dim {Xn.shape[1]}')

t_train = torch.from_numpy(Xtr).to(device)
t_val   = torch.from_numpy(Xval).to(device)


class AE(nn.Module):
    def __init__(self, in_dim=16, latent_dim=8): #Same architecture as best
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 512),    nn.LayerNorm(512),  nn.SiLU(),
            nn.Linear(512, 256),     nn.LayerNorm(256),  nn.SiLU(),
            nn.Linear(256, 128),     nn.LayerNorm(128),  nn.SiLU(),
            nn.Linear(128, 64),      nn.LayerNorm(64),   nn.SiLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LayerNorm(64),   nn.SiLU(),
            nn.Linear(64, 128),         nn.LayerNorm(128),  nn.SiLU(),
            nn.Linear(128, 256),        nn.LayerNorm(256),  nn.SiLU(),
            nn.Linear(256, 512),        nn.LayerNorm(512),  nn.SiLU(),
            nn.Linear(512, 1024),       nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, in_dim),    nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_one(latent_dim, slurm_start):
    """Train one AE. Resume from ae_d{D}_ckpt.pt if it exists. Stop early when
    cumulative slurm time exceeds WALL_BUDGET_S so we save before walltime kills us."""
    print(f'\n=== AE latent_dim={latent_dim} ===', flush=True)
    model = AE(latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    ckpt_path = f'ae_d{latent_dim}_ckpt.pt'
    hist_path = f'ae_d{latent_dim}_history.npz'
    start_epoch = 0
    losses_tr, losses_val = [], []
    if os.path.exists(ckpt_path):
        c = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(c['state_dict'])
        if 'opt_state' in c: opt.load_state_dict(c['opt_state'])
        start_epoch = int(c.get('epoch', 0))
        if os.path.exists(hist_path):
            h = np.load(hist_path)
            losses_tr  = list(h['train']); losses_val = list(h['val'])
        print(f'  resumed from epoch {start_epoch}', flush=True)
    # Resume = fine-tune at small LR. Full LR on resume kicks converged weights
    FINETUNE_LR = 5e-5 if start_epoch > 0 else LR
    for pg in opt.param_groups:
        pg['lr'] = FINETUNE_LR
    remaining = max(1, EPOCHS - start_epoch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining)
    if start_epoch >= EPOCHS:
        print(f'  already at target {EPOCHS} epochs, skipping', flush=True)
        return losses_val[-1] if losses_val else float('nan')

    n_tr = len(t_train)
    last_val = float('nan')
    for ep in range(start_epoch, EPOCHS):
        if time.time() - slurm_start > WALL_BUDGET_S:
            print(f'  walltime budget hit at epoch {ep}/{EPOCHS}, saving and exiting', flush=True)
            break
        model.train()
        perm = torch.randperm(n_tr, device=device)
        loss_sum = 0.0; n_seen = 0
        for s in range(0, n_tr, BATCH):
            ix = perm[s:s + BATCH]
            xb = t_train[ix]
            xh, _ = model(xb)
            loss = ((xh - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); n_seen += len(xb)
        sched.step()
        tr_mse = loss_sum / n_seen
        model.eval()
        with torch.no_grad():
            xhv, _ = model(t_val)
            val_mse = ((xhv - t_val) ** 2).mean().item() 
        losses_tr.append(tr_mse); losses_val.append(val_mse); last_val = val_mse
        if ep == start_epoch or (ep + 1) % 10 == 0:
            print(f'  ep {ep+1:>3d}/{EPOCHS}  train {tr_mse:.5f}  val {val_mse:.5f}', flush=True)
        # periodic checkpoint every 100 epochs, so timouts don't lose work
        if (ep + 1) % 100 == 0:
            torch.save({'state_dict': model.state_dict(),
                        'opt_state': opt.state_dict(),
                        'sched_state': sched.state_dict(),
                        'epoch': ep + 1,
                        'latent_dim': latent_dim,
                        'LO': LO, 'HI': HI}, ckpt_path)
            np.savez(hist_path, train=np.array(losses_tr), val=np.array(losses_val)) #history save
    final_epoch = start_epoch + (len(losses_tr) - (len(losses_tr) - (ep + 1 - start_epoch)))
    torch.save({'state_dict': model.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': sched.state_dict(),
                'epoch': ep + 1,
                'latent_dim': latent_dim,
                'LO': LO, 'HI': HI}, ckpt_path)
    np.savez(hist_path, train=np.array(losses_tr), val=np.array(losses_val))
    print(f'  saved {ckpt_path}  epoch={ep+1}  last val MSE {last_val:.5f}', flush=True)
    return last_val


t0 = time.time()
finals = {}
for D in LATENT_DIMS:
    finals[D] = train_one(D, slurm_start=t0)
print('\n=== Reconstruction MSE per latent dim (lower=better) ===')
for D, m in finals.items():
    print(f'  D={D:>2d}  val MSE = {m:.5f}')
print(f'total wall {time.time()-t0:.0f}s')
