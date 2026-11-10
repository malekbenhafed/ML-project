"""Training and evaluation of the steel plate fault models.

This module has no Streamlit dependency so it can be tested on its own.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scratch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path):
    """Read the CSV and strip stray spaces from the column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def split_features_labels(df):
    """Separate the input features (X) from the 7 fault labels (y)."""
    return df.drop(columns=LABEL_COLS), df[LABEL_COLS]


def build_models():
    """Return the three models, ready to be fitted.

    - Random Forest: one forest per fault (binary relevance). Trees compare
      thresholds, so they do not need scaled features.
    - KNN and MLP: distance/gradient based, so they need scaled features.
      The scaler lives inside a Pipeline, which guarantees it is fitted on
      the training data only (no data leakage).
    """
    return {
        "Random Forest": MultiOutputClassifier(
            RandomForestClassifier(random_state=RANDOM_STATE)
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3),
        ),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(100, 140),
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def evaluate(model, X_test, y_test):
    """Score a fitted model on the test set, the same way for every model.

    - mean_accuracy: accuracy of each fault, averaged over the 7 faults.
    - macro_f1: F1 of each fault, averaged (stricter on rare faults).
    - exact_match: share of plates where all 7 labels are correct.
    """
    pred = pd.DataFrame(
        model.predict(X_test), columns=LABEL_COLS, index=y_test.index
    )
    per_label = pd.DataFrame(
        {
            "accuracy": [accuracy_score(y_test[c], pred[c]) for c in LABEL_COLS],
            "f1": [
                f1_score(y_test[c], pred[c], zero_division=0) for c in LABEL_COLS
            ],
        },
        index=LABEL_COLS,
    )
    return {
        "mean_accuracy": float(per_label["accuracy"].mean()),
        "macro_f1": float(per_label["f1"].mean()),
        "exact_match": float((pred.values == y_test.values).all(axis=1).mean()),
        "per_label": per_label,
    }


def train_and_evaluate(df):
    """Split first, then fit and evaluate every model on the same split."""
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    models = build_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
    return models, results
