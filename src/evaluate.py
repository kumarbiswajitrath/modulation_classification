import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data import load_radioml, split
from src.model import AMCNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="results/best.pt")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, snr, mods = load_radioml(a.data)
    _, _, te = split(y, snr, a.seed)

    model = AMCNet(len(mods)).to(dev)
    model.load_state_dict(torch.load(a.weights, map_location=dev))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 1024):
            xb = torch.from_numpy(X[te[i:i + 1024]]).to(dev)
            preds.append(model(xb).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    yt, st = y[te], snr[te]
    print(f"overall test accuracy: {(preds == yt).mean():.4f}")

    # accuracy vs SNR
    snrs = sorted(set(st))
    accs = [(preds[st == s] == yt[st == s]).mean() for s in snrs]
    plt.figure(); plt.plot(snrs, accs, marker="o", label="CNN")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
    plt.title("Accuracy vs SNR"); plt.savefig("results/acc_vs_snr.png", dpi=150)

    # confusion matrix at high SNR
    m = st >= 10
    cm = confusion_matrix(yt[m], preds[m], normalize="true")
    plt.figure(figsize=(8, 7)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(mods)), mods, rotation=90); plt.yticks(range(len(mods)), mods)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix (SNR >= 10 dB)")
    plt.colorbar(); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
