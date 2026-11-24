# Vision Digits

Handwritten digit recognition with classic computer vision features and two classifiers, plus data augmentation and a prediction CLI that accepts any image.

## Approach
- Data: scikit-learn's bundled 8x8 digits (1,797 images), no download needed.
- Features: raw pixels + gradient magnitude + oriented-gradient histograms per 4x4 cell (a small HOG).
- Augmentation: training set tripled with random 1-pixel shifts.
- Models: RBF-SVM vs MLP (128-64). The best on the hold-out set is saved.

## Results (25% hold-out)
| model | accuracy |
|---|---|
| SVM (RBF) | 0.989 |
| MLP | 0.973 |

Most common confusion: 5 predicted as 9 (2 cases). See `report.json` after training.

## Run
```bash
pip install -r requirements.txt
python -m vision.train .                 # trains, writes digits.joblib + report.json
python -m vision.predict my_digit.png    # {'digit': 3, 'confidence': 0.97}
pytest -q
```
Images of any size work: they are inverted if drawn dark-on-light and downscaled to 8x8.

## Limits
8x8 images are tiny, so real handwriting photos work best when the digit is centered and fills the frame. A CNN on MNIST would be the natural next step.
