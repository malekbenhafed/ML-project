malekbenhafed
ML-project
Repository navigation
Code
Issues
Pull requests
Agents
Actions
Projects
Wiki
Security and quality
Insights
Settings
ML-project
/
streamlit_app.py
in
master

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing streamlit_app.py file contents
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
While the code is focused, press Alt+F1 for a menu of operations.
