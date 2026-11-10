"""Tests for model.py, run on small synthetic data (no CSV needed).

Run with:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model  # noqa: E402


def make_data(n=300, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(loc=10, scale=3, size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    classes = rng.integers(0, len(model.LABEL_COLS), size=n)
    X["f0"] += classes  # give the features some signal
    y = pd.DataFrame(
        np.eye(len(model.LABEL_COLS), dtype=int)[classes],
        columns=model.LABEL_COLS,
    )
    return pd.concat([X, y], axis=1)


def test_split_features_labels():
    df = make_data()
    X, y = model.split_features_labels(df)
    assert list(y.columns) == model.LABEL_COLS
    assert not set(model.LABEL_COLS) & set(X.columns)


def test_metrics_are_valid_and_comparable():
    _, results = model.train_and_evaluate(make_data())
    assert set(results) == {"Random Forest", "KNN", "MLP"}
    for res in results.values():
        assert 0 <= res["exact_match"] <= res["mean_accuracy"] <= 1
        assert 0 <= res["macro_f1"] <= 1
        assert res["per_label"].shape == (len(model.LABEL_COLS), 2)


def test_scaler_is_fitted_on_train_only():
    df = make_data()
    models, _ = model.train_and_evaluate(df)
    X, y = model.split_features_labels(df)
    X_train, _, _, _ = train_test_split(
        X, y, test_size=model.TEST_SIZE, random_state=model.RANDOM_STATE
    )
    scaler = models["KNN"][0]
    np.testing.assert_allclose(scaler.mean_, X_train.mean().values)
    assert not np.allclose(scaler.mean_, X.mean().values)


def test_predict_accepts_a_single_row():
    df = make_data()
    models, _ = model.train_and_evaluate(df)
    X, _ = model.split_features_labels(df)
    row = pd.DataFrame([X.mean().to_dict()], columns=X.columns)
    for m in models.values():
        pred = m.predict(row)[0]
        assert len(pred) == len(model.LABEL_COLS)
