import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import geometry_check as geom


class VAE(nn.Module):
    def __init__(self, in_dim=16, latent_dim=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), 
            nn.LayerNorm(1024), 
            nn.SiLU(),
            nn.Linear(1024, 512),    
            nn.LayerNorm(512),  
            nn.SiLU(),
            nn.Linear(512, 256),     
            nn.LayerNorm(256),  
            nn.SiLU(),
            nn.Linear(256, 128),     
            nn.LayerNorm(128),  
            nn.SiLU(),
            nn.Linear(128, 64),      
            nn.LayerNorm(64),   
            nn.SiLU(),
        )
        
        self.mu_head = nn.Linear(64, latent_dim)
        self.logvar_head = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  
            nn.LayerNorm(64),   
            nn.SiLU(),
            nn.Linear(64, 128),         
            nn.LayerNorm(128),  
            nn.SiLU(),
            nn.Linear(128, 256),        
            nn.LayerNorm(256),  
            nn.SiLU(),
            nn.Linear(256, 512),        
            nn.LayerNorm(512),  
            nn.SiLU(),
            nn.Linear(512, 1024),       
            nn.LayerNorm(1024), 
            nn.SiLU(),
            nn.Linear(1024, in_dim),    
            nn.Sigmoid(),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Start logvar near zero (sigma approx 1) so KL doesn't blow up early
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decoder(z), mu, logvar


def get_lr(ep, warmup, epochs, base_lr):
    """Cosine annealing with linear warmup."""
    if ep < warmup:
        return base_lr * (ep + 1) / warmup
    p = (ep - warmup) / max(1, epochs - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * p))


def get_beta(ep, epochs, beta_start, beta_end, anneal_frac):
    """Linear beta ramp for KL term."""
    anneal_end = int(epochs * anneal_frac)
    if ep >= anneal_end:
        return beta_end
    return beta_start + (beta_end - beta_start) * ep / anneal_end


def main(args):
    # Setup device and seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    # Load and prep data
    print(f"Loading data from {args.data_path}...")
    X = np.load(args.data_path).astype(np.float32)
    LO = geom.BOUNDS[:, 0].astype(np.float32)
    HI = geom.BOUNDS[:, 1].astype(np.float32)
    
    # Normalize
    Xn = (X - LO) / (HI - LO)
    
    # Train/val split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xn))
    n_tr = int(0.9 * len(Xn))
    
    Xtr, Xval = Xn[idx[:n_tr]], Xn[idx[n_tr:]]
    t_tr = torch.from_numpy(Xtr).to(device)
    t_val = torch.from_numpy(Xval).to(device)
    baseline_var = float(t_val.var(dim=0).mean())

    # Initialize model and optimizer
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    ckpt_path = f'vae_d{args.latent_dim}_prod_ckpt.pt'
    hist_path = f'vae_d{args.latent_dim}_prod_history.npz'

    best_val_recon = float('inf')
    best_ep = -1
    
    # Trackers
    history = {
        'train_loss': [], 'train_recon': [], 
        'train_kl': [], 'val_recon': []
    }
    
    t0 = time.time()
    n_full = len(t_tr)

    # Training loop
    for ep in range(args.epochs):
        lr_now = get_lr(ep, args.warmup, args.epochs, args.lr)
        beta_now = get_beta(ep, args.epochs, args.beta_start, args.beta_end, args.beta_anneal)
        
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now
            
        model.train()
        perm = torch.randperm(n_full, device=device)
        
        tr_loss = 0.0
        tr_recon = 0.0
        tr_kl = 0.0
        n_seen = 0
        
        for s in range(0, n_full, args.batch_size):
            ix = perm[s:s + args.batch_size]
            xb = t_tr[ix]
            
            xh, mu, logvar = model(xb, sample=True)
            
            recon = ((xh - xb) ** 2).mean()
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            loss = recon + beta_now * kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            batch_size = len(xb)
            tr_loss += loss.item() * batch_size
            tr_recon += recon.item() * batch_size
            tr_kl += kl.item() * batch_size
            n_seen += batch_size
            
        # Average metrics
        tr_loss /= n_seen
        tr_recon /= n_seen
        tr_kl /= n_seen

        # Validation
        model.eval()
        with torch.no_grad():
            xhv, _, _ = model(t_val, sample=False)
            val_recon = ((xhv - t_val) ** 2).mean().item()
            
        history['train_loss'].append(tr_loss)
        history['train_recon'].append(tr_recon)
        history['train_kl'].append(tr_kl)
        history['val_recon'].append(val_recon)

        # Save best model
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            best_ep = ep + 1
            torch.save({
                'state_dict': model.state_dict(), 
                'epoch': ep + 1,
                'val_recon': val_recon, 
                'latent_dim': args.latent_dim,
                'beta_final': beta_now, 
                'LO': LO, 
                'HI': HI,
                'baseline_var': baseline_var
            }, ckpt_path)

        # Logging
        if ep == 0 or (ep + 1) % 25 == 0:
            r2 = 1.0 - val_recon / baseline_var
            r2_best = 1.0 - best_val_recon / baseline_var
            print(f"Epoch {ep+1:04d}/{args.epochs} | LR: {lr_now:.2e} | Beta: {beta_now:.4f}")
            print(f"  Train -> Loss: {tr_loss:.4f} | Recon: {tr_recon:.5f} | KL: {tr_kl:.3f}")
            print(f"  Valid -> Recon: {val_recon:.5f} | R2: {r2:.4f}")
            print(f"  Best  -> Recon: {best_val_recon:.5f} (R2: {r2_best:.4f}) @ Epoch {best_ep}")
            
            # Checkpoint history periodically
            np.savez(hist_path, **history)

    # Final save and summary
    np.savez(hist_path, **history)
    r2_final = 1.0 - best_val_recon / baseline_var
    
    print("\n" + "="*30)
    print(f"Training Complete! Total time: {time.time() - t0:.0f}s")
    print(f"Best Val Recon: {best_val_recon:.5f} (R2: {r2_final:.4f}) at epoch {best_ep}")
    print("="*30)

    # Quick validity test
    model.eval()
    with torch.no_grad():
        z_sample = torch.randn(1000, args.latent_dim, device=device)
        x_decoded = model.decoder(z_sample).cpu().numpy()
        X_phys = LO + x_decoded * (HI - LO)
        
        n_valid = sum(geom.is_valid_geometry(x)[0] for x in X_phys[:200])
        print(f"N(0,I) sample validity = {n_valid}/200 ({n_valid/200*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE for geometry generation")
    
    # Model & Data args
    parser.add_argument('--data_path', type=str, default='training_X_valid_500k.npy')
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=100)
    
    # Beta annealing args
    parser.add_argument('--beta_start', type=float, default=1e-3)
    parser.add_argument('--beta_end', type=float, default=1e-2)
    parser.add_argument('--beta_anneal', type=float, default=0.5, help='Fraction of epochs to anneal beta')

    args = parser.parse_args()
    main(args)
