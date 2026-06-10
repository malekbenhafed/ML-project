import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE TITLE
# ==========================================

st.title("🔩 Steel Plate Fault Detection App")
st.write("A machine learning web app that detects faults in steel plates.")

# ==========================================
# LOAD DATASET
# ==========================================
import os

st.write("Current files:")
st.write(os.listdir())
df = pd.read_csv("Steel_Plates_Faults.csv")
df.columns = df.columns.str.strip()

# Target columns
label_cols = [
    'Pastry',
    'Z_Scratch',
    'K_Scratch',
    'Stains',
    'Dirtiness',
    'Bumps',
    'Other_Faults'
]

# Features and labels
X = df.drop(columns=label_cols)
y = df[label_cols]

# ==========================================
# DATA PREPROCESSING
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODELS
# ==========================================

rf_models = {}

for label in label_cols:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train[label])
    rf_models[label] = model

# ==========================================
# MLP MODEL
# ==========================================

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 140),
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

# ==========================================
# KNN MODEL
# ==========================================

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ==========================================
# MODEL ACCURACY
# ==========================================

rf_acc = round(
    sum(
        accuracy_score(
            y_test[label],
            rf_models[label].predict(X_test)
        )
        for label in label_cols
    ) / len(label_cols),
    4
)

mlp_acc = round(
    accuracy_score(y_test, mlp.predict(X_test)),
    4
)

knn_acc = round(
    accuracy_score(y_test, knn.predict(X_test)),
    4
)

# ==========================================
# SIDEBAR INPUT
# ==========================================

st.sidebar.header("⚙️ Input Features")

input_data = {}

for col in X.columns:
    input_data[col] = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])

# Scale input
input_scaled = scaler.transform(input_df)

# ==========================================
# SHOW INPUT
# ==========================================

st.subheader("📋 Input Data")
st.write(input_df)

# ==========================================
# PREDICTIONS
# ==========================================

st.subheader("🔍 Predicted Faults")

results = {}

for label, model in rf_models.items():
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        results[label] = "✅ Detected"
    else:
        results[label] = "❌ Not Detected"

results_df = pd.DataFrame(
    results.items(),
    columns=["Fault Type", "Prediction"]
)

st.table(results_df)

# ==========================================
# ACCURACY CHART
# ==========================================

st.subheader("📊 Model Accuracy Comparison")

model_names = [
    "KNN",
    "MLP",
    "Random Forest"
]

accuracies = [
    knn_acc,
    mlp_acc,
    rf_acc
]

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(model_names, accuracies)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

st.pyplot(fig)

# ==========================================
# ACCURACY TABLE
# ==========================================

st.subheader("📈 Accuracy Scores")

accuracy_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies
})

st.dataframe(accuracy_df)
