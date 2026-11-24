"""Handwritten digit recognition: HOG-like gradient features + SVM vs a small MLP,
with data augmentation (random shifts), confusion matrix and a saved model."""
import json
import sys

import joblib
import numpy as np
from scipy.ndimage import shift
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


def gradient_features(X: np.ndarray) -> np.ndarray:
    """Pixels plus Sobel-style gradient magnitude and 4-bin orientation histograms per 4x4 cell."""
    imgs = X.reshape(-1, 8, 8)
    gx = np.zeros_like(imgs); gy = np.zeros_like(imgs)
    gx[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
    gy[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * 4
    ori = np.minimum(ang.astype(int), 3)
    feats = [X, mag.reshape(len(X), -1)]
    for r in (0, 4):
        for c in (0, 4):
            for o in range(4):
                m = (ori[:, r:r + 4, c:c + 4] == o) * mag[:, r:r + 4, c:c + 4]
                feats.append(m.sum(axis=(1, 2))[:, None])
    return np.hstack(feats)


def augment(X: np.ndarray, y: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = [X]
    for _ in range(2):
        out.append(np.stack([shift(x.reshape(8, 8), rng.integers(-1, 2, 2), order=0, cval=0).ravel() for x in X]))
    return np.vstack(out), np.concatenate([y] * 3)


def main(out: str = ".") -> dict:
    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, stratify=d.target, random_state=1)
    Xa, ya = augment(Xtr, ytr)
    feat = FunctionTransformer(gradient_features)
    models = {
        "svm_rbf": make_pipeline(feat, StandardScaler(), SVC(C=10, gamma="scale", probability=True)),
        "mlp": make_pipeline(feat, StandardScaler(), MLPClassifier((128, 64), max_iter=400, random_state=0)),
    }
    res = {}
    for n, m in models.items():
        m.fit(Xa, ya)
        res[n] = round(accuracy_score(yte, m.predict(Xte)), 4)
    best = max(res, key=res.get)
    model = models[best]
    cm = confusion_matrix(yte, model.predict(Xte))
    worst = [(int(i), int(j), int(cm[i, j])) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    worst.sort(key=lambda t: -t[2])
    report = {"accuracy": res, "selected": best, "top_confusions(true,pred,count)": worst[:5]}
    joblib.dump(model, f"{out}/digits.joblib")
    json.dump(report, open(f"{out}/report.json", "w"), indent=2)
    return report


if __name__ == "__main__":
    from vision.train import main as _main  # import by module path so pickles resolve outside __main__
    print(json.dumps(_main(sys.argv[1] if len(sys.argv) > 1 else "."), indent=2))
