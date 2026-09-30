import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_radioml, split
from src.model import AMCNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to RML2016.10a_dict.pkl")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, snr, mods = load_radioml(a.data)
    tr, va, _ = split(y, snr, a.seed)
    to_t = lambda i: TensorDataset(torch.from_numpy(X[i]), torch.from_numpy(y[i]))
    tr_dl = DataLoader(to_t(tr), batch_size=a.batch, shuffle=True)
    va_dl = DataLoader(to_t(va), batch_size=a.batch)

    model = AMCNet(len(mods)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0

    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                correct += (model(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
        acc = correct / len(va)
        print(f"epoch {ep:02d}  val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "results/best.pt")
    print(f"best val acc: {best:.4f}")


if __name__ == "__main__":
    main()
