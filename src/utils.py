
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


DATA_DEFAULT_RED = os.path.join("..", "data", "raw", "winequality-red.csv")
DATA_DEFAULT_WHITE = os.path.join("..", "data", "raw", "winequality-white.csv")


def load_wine_data(
    red_path: str = DATA_DEFAULT_RED,
    white_path: str = DATA_DEFAULT_WHITE,
    uploaded_red: Optional[Any] = None,
    uploaded_white: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Loads and concatenates red/white wine datasets.
    If uploaded files are provided (Streamlit UploadedFile), they override disk paths.
    """
    if uploaded_red is not None:
        red = pd.read_csv(uploaded_red, sep=";")
    else:
        red = pd.read_csv(red_path, sep=";")

    if uploaded_white is not None:
        white = pd.read_csv(uploaded_white, sep=";")
    else:
        white = pd.read_csv(white_path, sep=";")

    red["wine type"] = "red"
    white["wine type"] = "white"

    data = pd.concat([red, white], axis=0).reset_index(drop=True)
    return data


def add_target_quality_binary(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    out = df.copy()
    out["quality_binary"] = (out["quality"] >= threshold).astype(int)
    return out


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["quality_binary"])
    y = df["quality_binary"]
    return X, y


def make_smote_pipeline(model_name: str, random_state: int = 42) -> ImbPipeline:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
    ]
    models = {
        "Logistic Regression + SMOTE": LogisticRegression(max_iter=500, random_state=random_state),
        "k-NN + SMOTE": KNeighborsClassifier(),
        "Decision Tree + SMOTE": DecisionTreeClassifier(random_state=random_state),
        "Naive Bayes + SMOTE": GaussianNB(),
        "SVM (RBF) + SMOTE": SVC(probability=True, kernel="rbf", random_state=random_state),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model_name: {model_name}")
    return ImbPipeline(base_steps + [("model", models[model_name])])


@dataclass
class EvalArtifacts:
    metrics: Dict[str, float]
    cm: np.ndarray
    roc: Tuple[np.ndarray, np.ndarray, np.ndarray]  # fpr, tpr, thresholds
    pr: Tuple[np.ndarray, np.ndarray, np.ndarray]   # precision, recall, thresholds


def evaluate_binary(
    pipe: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvalArtifacts:
    y_pred = pipe.predict(X_test)

    # Probabilities for ROC-AUC (if available)
    y_proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, thr = roc_curve(y_test, y_proba)
            prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)
        except Exception:
            fpr, tpr, thr = np.array([]), np.array([]), np.array([])
            prec, rec, pr_thr = np.array([]), np.array([]), np.array([])
    else:
        metrics["roc_auc"] = float("nan")
        fpr, tpr, thr = np.array([]), np.array([]), np.array([])
        prec, rec, pr_thr = np.array([]), np.array([]), np.array([])

    cm = confusion_matrix(y_test, y_pred)
    return EvalArtifacts(metrics=metrics, cm=cm, roc=(fpr, tpr, thr), pr=(prec, rec, pr_thr))


def train_one_model(
    df: pd.DataFrame,
    model_name: str,
    threshold: int = 7,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ImbPipeline, EvalArtifacts, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    df2 = add_target_quality_binary(df, threshold=threshold)
    X, y = split_xy(df2)

    # Split BEFORE any scaling/SMOTE (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = make_smote_pipeline(model_name, random_state=random_state)
    pipe.fit(X_train, y_train)

    artifacts = evaluate_binary(pipe, X_test, y_test)
    return pipe, artifacts, (X_train, X_test, y_train, y_test)
