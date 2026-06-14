# 🔩 Steel Plate Fault Detection App

A machine learning web app that detects faults in steel plates.  
Users adjust input feature values using interactive sliders, and the app instantly predicts the fault type:  
**Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, or Other_Faults**  
using a trained **Random Forest** model built on the UCI Steel Plates Faults dataset.

## 🚀 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ML-project.streamlit.app/)

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)
- 1941 samples, 27 input features, 7 fault categories
- Multi-label classification problem

## 🧠 Models Used

| Model | Notes |
|-------|-------|
| Random Forest | Main model used in the app |
| KNN | Compared during training |
| MLPClassifier | Compared during training |

## ⚙️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas / NumPy
