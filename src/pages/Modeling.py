
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_wine_data, add_target_quality_binary, train_one_model

st.set_page_config(page_title="Modélisation • Wine Quality", page_icon="🤖", layout="wide")
st.title("🤖 Modélisation — Pipelines (StandardScaler + SMOTE + Modèle)")

st.markdown(
"""
Sélectionne un modèle, entraîne-le **sans fuite de données** :

- Split Train/Test d'abord (données brutes)
- `StandardScaler` + `SMOTE` dans le pipeline, entraînés uniquement sur `X_train`
"""
)

st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox(
    "Modèle",
    [
        "Logistic Regression + SMOTE",
        "k-NN + SMOTE",
        "Decision Tree + SMOTE",
        "Naive Bayes + SMOTE",
        "SVM (RBF) + SMOTE"
    ],
    index=0
)
threshold = st.sidebar.slider("Seuil qualité → classe 1 si quality ≥", 5, 8, 7, 1)
test_size = st.sidebar.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)

@st.cache_data(show_spinner=False)
def get_df(thr):
    df = load_wine_data()
    df = add_target_quality_binary(df, threshold=thr)
    return df

df = get_df(threshold)

train_btn = st.button("🚀 Entraîner le modèle", type="primary")

if train_btn:
    with st.spinner("Entraînement en cours..."):
        pipe, artifacts, splits = train_one_model(
            df, model_name=model_name, threshold=threshold, test_size=test_size
        )
    st.session_state["last_model_name"] = model_name
    st.session_state["last_pipe"] = pipe
    st.session_state["last_artifacts"] = artifacts

if "last_artifacts" in st.session_state:
    artifacts = st.session_state["last_artifacts"]
    st.success(f"Modèle entraîné : {st.session_state['last_model_name']}")

    m = artifacts.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1", f"{m['f1']:.3f}")
    c5.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")

    st.caption("Va dans l’onglet Évaluation pour les courbes ROC/PR et la matrice de confusion.")
else:
    st.info("Clique sur **Entraîner le modèle** pour afficher les métriques.")
