# 🔩 Steel Plate Fault Detection App

A machine learning web app that detects faults in steel plates.
Adjust the input features with the sliders in the sidebar and the app predicts which of the 7 fault types are present: **Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other_Faults**.

## 🚀 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ML-project.streamlit.app/)

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)
- 1941 samples, 27 input features, 7 fault labels
- Every plate in this file carries exactly one fault, but the 7 labels are modelled independently (multi-label setup)
- Class sizes are unbalanced: from 55 plates (Dirtiness) to 673 (Other_Faults)

## 🧠 Approach

| Model | How it is built |
|-------|-----------------|
| Random Forest | One forest per fault. No scaling (trees compare thresholds, so scale does not matter) |
| KNN (k=3) | `StandardScaler` + KNN, inside a `Pipeline` |
| MLP (100, 140) | `StandardScaler` + MLP, inside a `Pipeline` |

Design choices:

- **Split first, scale after.** The data is split 80/20 (`random_state=42`) before anything is learned from it. The scaler lives inside a `Pipeline`, so it only ever sees the training set: no data leakage.
- **Same metrics for every model.** The three models are compared on the same test set with the same three scores:
  - `mean_accuracy`: accuracy of each fault, averaged over the 7 faults
  - `macro_f1`: F1 of each fault, averaged (stricter on rare faults)
  - `exact_match`: share of plates where all 7 labels are correct
- **Training is cached.** Streamlit re-runs the script on every slider move, so the models are trained once with `st.cache_resource`.

## 📈 Results

Held-out test set (20% of the data, 389 plates):

| Model | mean_accuracy | macro_f1 | exact_match |
|-------|--------------:|---------:|------------:|
| Random Forest | 0.937 | 0.773 | 0.666 |
| KNN | 0.923 | 0.708 | 0.717 |
| MLP | 0.934 | 0.743 | 0.715 |

How to read this:

- The three models are close on `mean_accuracy`. This score is flattered by rare faults: predicting "absent" is almost always right for them. `macro_f1` is the more honest score.
- Random Forest has the best `macro_f1`, but the lowest `exact_match`: since each fault is predicted independently, it can return zero or several faults for one plate.
- The weakest faults are **Bumps** and **Pastry** (F1 around 0.57-0.59).

## ⚠️ Limitations

- A single train/test split: the scores above move by a few points with a different split. Cross-validation would give a more reliable estimate.
- No hyperparameter tuning.
- Each plate has exactly one fault in this dataset, but the models do not use that constraint (a single multi-class model could).
- The sliders let you build feature combinations that may never occur on a real plate.

## 🗂️ Project structure

```
streamlit_app.py     # the web app (interface only)
model.py             # data loading, models, evaluation
tests/test_model.py  # tests on synthetic data
Steel_Plates_Faults.csv
requirements.txt
```

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the tests:

```bash
pip install pytest
pytest
```

## 🛠️ Tech Stack

- Python
- scikit-learn
- Streamlit
- pandas / NumPy / Matplotlib
