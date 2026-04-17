import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCNAlign(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.proj = nn.Linear(in_dim * (num_hops + 1), out_dim)
        self.recon = nn.Linear(out_dim, in_dim)
        self.transform = nn.Linear(out_dim, out_dim, bias=False)

    def encode(self, A, X):
        feats = [X]
        H = X
        for _ in range(self.num_hops):
            H = A @ H
            feats.append(H)
        H_cat = torch.cat(feats, dim=1)
        Z = self.proj(H_cat)
        Z = F.normalize(Z, dim=1)
        return Z

    def forward(self, As, Xs, At, Xt):
        Zs = self.encode(As, Xs)
        Zt = self.encode(At, Xt)
        Zs_t = self.transform(Zs)
        Xs_rec = self.recon(Zs)
        Xt_rec = self.recon(Zt)
        return Zs_t, Zt, Xs_rec, Xt_rec

def mutual_nn_pairs(Zs, Zt):
    sim = F.cosine_similarity(Zs[:, None, :], Zt[None, :, :], dim=-1)
    s2t = sim.argmax(dim=1)
    t2s = sim.argmax(dim=0)
    pairs = []
    for i in range(len(s2t)):
        j = s2t[i].item()
        if t2s[j].item() == i:
            pairs.append((i, j))
    return pairs, sim

def train_walign_epoch(model, As, Xs, At, Xt, optimizer, lambda_rec=1.0, lambda_align=1.0):
    model.train()
    optimizer.zero_grad()

    Zs, Zt, Xs_rec, Xt_rec = model(As, Xs, At, Xt)
    pairs, sim = mutual_nn_pairs(Zs, Zt)

    if len(pairs) > 0:
        idx_s = torch.tensor([p[0] for p in pairs], device=Zs.device)
        idx_t = torch.tensor([p[1] for p in pairs], device=Zt.device)
        align_loss = F.mse_loss(Zs[idx_s], Zt[idx_t])
    else:
        align_loss = torch.tensor(0.0, device=Zs.device)

    rec_loss = F.mse_loss(Xs_rec, Xs) + F.mse_loss(Xt_rec, Xt)

    loss = lambda_align * align_loss + lambda_rec * rec_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "align_loss": align_loss.item(),
        "rec_loss": rec_loss.item()
    }