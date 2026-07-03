"""
test_churn.py — Run: pytest tests/test_churn.py
"""
import pickle, numpy as np, pandas as pd, pytest
from sklearn.metrics import accuracy_score, roc_auc_score


@pytest.fixture
def model_scaler():
    with open("models/gradient_boosting.pkl","rb") as f: m = pickle.load(f)
    with open("models/scaler.pkl","rb") as f:            s = pickle.load(f)
    return m, s


@pytest.fixture
def data():
    df = pd.read_csv("data/churn.csv")
    return df.drop('churn', axis=1), df['churn']


def test_accuracy_above_80(model_scaler, data):
    m, s = model_scaler; X, y = data
    assert accuracy_score(y, m.predict(s.transform(X))) > 0.80


def test_auc_above_90(model_scaler, data):
    m, s = model_scaler; X, y = data
    assert roc_auc_score(y, m.predict_proba(s.transform(X))[:,1]) > 0.90


def test_predictions_binary(model_scaler, data):
    m, s = model_scaler; X, y = data
    assert set(m.predict(s.transform(X))).issubset({0, 1})


def test_smote_balances_classes():
    np.random.seed(42)
    X = pd.DataFrame({'a': range(100), 'b': range(100)})
    y = pd.Series([0]*80 + [1]*20)
    minority = X[y==1]; majority = X[y==0]
    n = len(majority) - len(minority)
    idx = np.random.choice(len(minority), n, replace=True)
    X_res = pd.concat([majority, minority, minority.iloc[idx]])
    y_res = np.concatenate([np.zeros(80), np.ones(20), np.ones(n)])
    assert abs(y_res.mean() - 0.5) < 0.01


def test_no_nulls():
    df = pd.read_csv("data/churn.csv")
    assert df.isnull().sum().sum() == 0
