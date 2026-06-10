import streamlit as st

st.title('🔩 Steel Plate Fault Detection App')

st.write('A 🤖 machine learning web app that detects faults in steel plates')
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Steel_Plates_Faults.csv")
df.columns = df.columns.str.strip()

label_cols = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
X = df.drop(columns=label_cols)
y = df[label_cols]

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train one model per label
models = {}
for label in label_cols:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train[label])
    models[label] = rf

# UI
st.title("🔩 Steel Plate Fault Detection App")
st.write("A 🤖 machine learning web app that detects faults in steel plates using Random Forest.")

st.sidebar.header("⚙️ Input Features")

def user_input():
    data = {}
    for col in X.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    return pd.DataFrame(data, index=[0])

input_df = user_input()
input_scaled = scaler.transform(input_df)

st.subheader("📋 Input Data")
st.write(input_df)

st.subheader("🔍 Predicted Faults")
results = {}
for label, model in models.items():
    pred = model.predict(input_scaled)[0]
    results[label] = "✅ Detected" if pred == 1 else "❌ Not Detected"

st.table(pd.DataFrame(results, index=["Prediction"]).T)
