# Crowd Density Image Classification Using CNN and Transfer Learning

A machine learning project that classifies crowd density in images into three categories — **Low**, **Medium**, and **High** — using the ShanghaiTech dataset. Implements three models (Custom CNN, MobileNetV2, ResNet50) with Grad-CAM and SHAP explainability.

> **⚠️ Note on notebook rendering:** GitHub does not render large Jupyter notebooks fully in the browser. Sections 10 (SHAP) and 11 (Final Summary) may not be visible when viewing the notebook on GitHub. To see the complete notebook with all outputs, please download the file and open it locally, or view it directly on [Kaggle](https://www.kaggle.com) where it was originally run.
 
---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|---|---|---|---|---|---|
| Custom CNN | 68.9% | 72.2% | 68.9% | 69.3% | 405,059 |
| MobileNetV2 | 78.3% | 78.1% | 78.3% | 77.1% | 2,586,691 |
| ResNet50 | **78.9%** | **79.2%** | **78.9%** | **77.8%** | 24,113,027 |

---

## Dataset

**ShanghaiTech Crowd Counting Dataset**
- Source: https://www.kaggle.com/datasets/tthien/shanghaitech
- 1,198 images total (Part A + Part B)
- Original task: crowd counting (regression)
- Adapted for classification by converting head counts to 3 density classes:

| Class | Head Count Range | Images |
|---|---|---|
| Low | 0 – 50 people | 166 |
| Medium | 51 – 300 people | 718 |
| High | 301+ people | 314 |

Part A contains dense urban crowd scenes (avg ~501 people/image). Part B contains sparser Shanghai street scenes (avg ~124 people/image).

---

## Project Structure

```
crowd-density-classification/
├── cnn-crowd-density.ipynb     ← Main Kaggle notebook (all code)
├── figures/                    ← Output figures (PDF)
│   ├── fig1a_class_distribution.pdf
│   ├── fig1b_sample_images.pdf
│   ├── fig3_training_curves.pdf
│   ├── fig4_confusion_matrices.pdf
│   ├── fig5_gradcam_correct.pdf
│   ├── fig6_gradcam_misclassified.pdf
│   └── fig7_shap_explanations.pdf
├── model_comparison.csv        ← Final metrics table
└── README.md
```

---

## Models

**Custom CNN** — 4-block convolutional network trained from scratch with class weighting and sparse categorical cross-entropy. Demonstrates baseline performance on a small dataset.

**MobileNetV2** — Lightweight pretrained model fine-tuned in two phases: frozen base (10 epochs) then top-30-layer fine-tuning (20 epochs).

**ResNet50** — Deeper pretrained model with same two-phase training strategy. Achieves best overall accuracy.

---

## Explainability

**Grad-CAM** — Gradient-weighted Class Activation Maps highlight which image regions each model focuses on when making predictions. Applied to both correctly classified and misclassified examples.

**SHAP** — GradientExplainer shows pixel-level feature importance, revealing which pixels positively influenced each model's prediction.

---

## How to Run

### On Kaggle (recommended)

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Add the dataset: click **Add Data** → search `tthien/shanghaitech` → add it
3. Upload `cnn-crowd-density.ipynb` via **File → Import Notebook**
4. Set accelerator to **GPU T4** under **Settings → Accelerator**
5. Click **Run All** — full pipeline takes approximately 15–20 minutes

### Requirements

All dependencies are pre-installed in the Kaggle environment. If running locally:

```bash
pip install tensorflow==2.19.0
pip install numpy pandas matplotlib seaborn scikit-learn
pip install opencv-python scipy shap
```

### Dataset path

The notebook expects the dataset at:
```
/kaggle/input/datasets/tthien/shanghaitech/ShanghaiTech/
```

If running locally, update `DATA_ROOT` in Section 1 to your local path:
```python
DATA_ROOT = Path('/your/local/path/ShanghaiTech')
```

---

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Imports, constants, dataset path |
| 2 | Dataset loading, `.mat` annotation parsing, label generation |
| 3 | EDA — class distribution, sample images |
| 4 | Data pipeline (tf.data), train/val/test split, augmentation |
| 5 | Model definitions (Custom CNN, MobileNetV2, ResNet50) |
| 6 | Model training with callbacks |
| 7 | Training curves (accuracy + loss) |
| 8 | Evaluation — confusion matrices, classification reports, comparison table |
| 9 | Grad-CAM visualizations (correct + misclassified) |
| 10 | SHAP explanations |
| 11 | Final summary |

---

## Key Design Decisions

- **Count-to-class conversion**: thresholds (0–50, 51–300, 301+) chosen to reflect crowd management literature definitions of low/medium/high density
- **Class imbalance**: handled via `compute_class_weight` with `sparse_categorical_crossentropy` for the custom CNN
- **Two-phase transfer learning**: base frozen first to train the classification head, then top layers unfrozen for fine-tuning
- **Separate data pipelines**: one-hot labels for transfer learning models, sparse integer labels for custom CNN (required for class weighting in tf.data)
