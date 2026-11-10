from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model import LABEL_COLS, load_data, split_features_labels, train_and_evaluate

DATA_PATH = Path(__file__).parent / "Steel_Plates_Faults.csv"

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Steel Fault Detection", layout="wide")

st.title("🔩 Steel Plate Fault Detection App")
st.write("A machine learning web app that detects faults in steel plates.")


# =========================
# DATA AND MODELS (cached)
# =========================
# Streamlit re-runs this script on every slider move. Without caching, the
# models would be retrained each time.
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_resource
def get_models():
    return train_and_evaluate(get_data())


df = get_data()
X, _ = split_features_labels(df)

with st.spinner("Training the models (first load only)..."):
    models, results = get_models()

# =========================
# MODEL COMPARISON
# =========================
st.subheader("📊 Model comparison (20% held-out test set)")

summary = pd.DataFrame(
    {
        name: {
            "mean_accuracy": res["mean_accuracy"],
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
        }
        for name, res in results.items()
    }
).T

col_table, col_chart = st.columns(2)

with col_table:
    st.dataframe(summary.round(3))
    st.caption(
        "**mean_accuracy**: accuracy of each fault, averaged over the 7 faults. "
        "**macro_f1**: F1 of each fault, averaged; stricter on rare faults. "
        "**exact_match**: share of plates where all 7 labels are correct. "
        "All three models are scored the same way on the same split."
    )

with col_chart:
    fig, ax = plt.subplots()
    summary[["mean_accuracy", "macro_f1"]].plot.bar(ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    st.pyplot(fig)

with st.expander("Per-fault scores (Random Forest)"):
    st.dataframe(results["Random Forest"]["per_label"].round(3))

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("⚙️ Input Features")

input_data = {}

for col in X.columns:
    low, high = float(df[col].min()), float(df[col].max())
    if low == high:
        # A slider needs min < max; a constant feature has nothing to adjust.
        input_data[col] = low
        st.sidebar.caption(f"{col}: constant ({low})")
    else:
        input_data[col] = st.sidebar.slider(col, low, high, float(df[col].mean()))

input_df = pd.DataFrame([input_data], columns=X.columns)

# =========================
# PREDICTION
# =========================
st.subheader("🔍 Predictions")

model_name = st.selectbox(
    "Model used for the prediction", list(models), index=0
)
pred = models[model_name].predict(input_df)[0]

st.table(
    pd.DataFrame(
        {
            "Fault": LABEL_COLS,
            "Prediction": [
                "✅ Detected" if p == 1 else "❌ Not detected" for p in pred
            ],
        }
    )
)
