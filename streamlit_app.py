import streamlit as st

st.title('🔩 Steel Plate Fault Detection App')

st.write('A 🤖 machine learning web app that detects faults in steel plates')
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/malekbenhafed/project/master/Steel_Plates_Faults.csv")
df.columns = df.columns.str.strip()

label_cols = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
X = df.drop(columns=label_cols)
y = df[label_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
rf_models = {}
for label in label_cols:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train[label])
    rf_models[label] = rf

mlp = MLPClassifier(hidden_layer_sizes=(100, 140), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Accuracies
rf_acc = round(sum([accuracy_score(y_test[label], rf_models[label].predict(X_test)) for label in label_cols]) / len(label_cols), 4)
mlp_acc = round(accuracy_score(y_test, mlp.predict(X_test)), 4)
knn_acc = round(accuracy_score(y_test, knn.predict(X_test)), 4)

# UI
st.title("🔩 Steel Plate Fault Detection App")
st.write("A 🤖 machine learning web app that detects faults in steel plates.")

# Sidebar
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

# Input data
st.subheader("📋 Input Data")
st.write(input_df)

# Predictions
st.subheader("🔍 Predicted Faults (Random Forest)")
results = {}
for label, model in rf_models.items():
    pred = model.predict(input_scaled)[0]
    results[label] = "✅ Detected" if pred == 1 else "❌ Not Detected"
st.table(pd.DataFrame(results, index=["Prediction"]).T)

# Model comparison
st.subheader("📊 Model Accuracy Comparison")
model_names = ['KNN', 'MLP', 'Random Forest']
accuracies = [knn_acc, mlp_acc, rf_acc]
fig, ax = plt.subplots()
ax.bar(model_names, accuracies, color='skyblue')
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
st.pyplot(fig)

st.subheader("📈 Accuracy Scores")
st.write(pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies
}))
 
