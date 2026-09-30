import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_radioml(path):
    """Load RadioML 2016.10A (RML2016.10a_dict.pkl). Returns X (N,2,128), y, snr, class names."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d})
    X, y, snr = [], [], []
    for (mod, s), arr in d.items():
        X.append(arr)
        y += [mods.index(mod)] * len(arr)
        snr += [s] * len(arr)
    return np.vstack(X).astype(np.float32), np.array(y), np.array(snr), mods


def split(y, snr, seed=42):
    """60/20/20 split, stratified by (modulation, SNR). Same seed -> same split everywhere."""
    key = y * 100 + (snr + 20)
    idx = np.arange(len(y))
    train, tmp = train_test_split(idx, test_size=0.4, stratify=key, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=key[tmp], random_state=seed)
    return train, val, test
