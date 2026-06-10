import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Steel Fault Detection", layout="wide")

st.title("🔩 Steel Plate Fault Detection App")
st.write("A machine learning web app that detects faults in steel plates.")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Steel_Plates_Faults.csv")
df.columns = df.columns.str.strip()

# =========================
# LABELS
# =========================
label_cols = [
    'Pastry', 'Z_Scratch', 'K_Scratch',
    'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'
]

X = df.drop(columns=label_cols)
y = df[label_cols]

# =========================
# PREPROCESSING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# MODELS (SAFE VERSION)
# =========================
rf_models = {}

for label in label_cols:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train[label])
    rf_models[label] = model

# =========================
# ACCURACY
# =========================
rf_acc = sum(
    accuracy_score(y_test[label], rf_models[label].predict(X_test))
    for label in label_cols
) / len(label_cols)

st.subheader("📊 Model Accuracy")
st.write("Random Forest Accuracy:", round(rf_acc, 4))

# =========================
# SIDEBAR INPUT
# =========================
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
input_scaled = scaler.transform(input_df)

# =========================
# PREDICTION
# =========================
st.subheader("🔍 Predictions")

results = {}

for label, model in rf_models.items():
    pred = model.predict(input_scaled)[0]
    results[label] = "✅ Detected" if pred == 1 else "❌ Not Detected"

st.table(pd.DataFrame(results.items(), columns=["Fault", "Prediction"]))

# =========================
# CHART
# =========================
mlp = MLPClassifier(hidden_layer_sizes=(100, 140), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

st.subheader("📈 Accuracy Chart")
fig, ax = plt.subplots()
ax.bar(['KNN', 'MLP', 'Random Forest'], [knn_acc, mlp_acc, rf_acc], color='skyblue')
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison")
st.pyplot(fig)
