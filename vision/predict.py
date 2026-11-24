"""Classify a digit image: python -m vision.predict image.png  (any size, dark digit on light or light on dark)."""
import sys

import joblib
import numpy as np
from PIL import Image

from . import train  # noqa: F401  (needed so the pickled feature function resolves)


def to_8x8(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=float)
    if a.mean() > 127:  # dark ink on white paper -> invert to match the dataset
        a = 255 - a
    img = Image.fromarray(a.astype("uint8")).resize((8, 8), Image.LANCZOS)
    return (np.asarray(img, dtype=float) / 255 * 16).reshape(1, -1)


if __name__ == "__main__":
    model = joblib.load("digits.joblib")
    x = to_8x8(sys.argv[1])
    p = model.predict_proba(x)[0]
    print({"digit": int(p.argmax()), "confidence": round(float(p.max()), 3)})
