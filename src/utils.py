from __future__ import annotations

from pathlib import Path
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DEFAULT_RED = DATA_DIR / "winequality-red.csv"
DATA_DEFAULT_WHITE = DATA_DIR / "winequality-white.csv"


def _resolve_data_file(path: Optional[str | Path], default: Path, label: str) -> Path:
    """Return a concrete path, raising a clear error if the file is missing."""
    file_path = Path(path) if path is not None else default
    if not file_path.exists():
        raise FileNotFoundError(f"{label} dataset not found at {file_path}")
    return file_path


def load_wine_data(
    red_path: Optional[str | Path] = None,
    white_path: Optional[str | Path] = None,
    uploaded_red: Optional[Any] = None,
    uploaded_white: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Loads and concatenates red/white wine datasets.
    Uploaded Streamlit files override disk paths when provided.
    """

    if uploaded_red is not None:
        red = pd.read_csv(uploaded_red, sep=";")
    else:
        red_file = _resolve_data_file(red_path, DATA_DEFAULT_RED, "Red wine")
        red = pd.read_csv(red_file, sep=";")

    if uploaded_white is not None:
        white = pd.read_csv(uploaded_white, sep=";")
    else:
        white_file = _resolve_data_file(white_path, DATA_DEFAULT_WHITE, "White wine")
        white = pd.read_csv(white_file, sep=";")

    red["wine type"] = "red"
    white["wine type"] = "white"

    data = pd.concat([red, white], axis=0).reset_index(drop=True)
    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that handles numeric + categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No valid features found to build the preprocessing pipeline.")

    return ColumnTransformer(transformers)


def add_target_quality_binary(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    out = df.copy()
    out["quality_binary"] = (out["quality"] >= threshold).astype(int)
    return out


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["quality_binary"])
    y = df["quality_binary"]
    return X, y


def make_smote_pipeline(
    model_name: str,
    preprocessor: ColumnTransformer,
    random_state: int = 42,
) -> ImbPipeline:
    base_steps = [
        ("preprocess", preprocessor),
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

    preprocessor = build_preprocessor(X_train)
    pipe = make_smote_pipeline(
        model_name,
        preprocessor=preprocessor,
        random_state=random_state,
    )
    pipe.fit(X_train, y_train)

    artifacts = evaluate_binary(pipe, X_test, y_test)
    return pipe, artifacts, (X_train, X_test, y_train, y_test)
