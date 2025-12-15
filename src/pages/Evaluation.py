
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Évaluation • Wine Quality", page_icon="🧪", layout="wide")
st.title("🧪 Évaluation — Confusion matrix, ROC, PR")

if "last_artifacts" not in st.session_state:
    st.warning("Aucun modèle entraîné. Va d’abord dans **Modélisation** puis entraîne un modèle.")
    st.stop()

art = st.session_state["last_artifacts"]
m = art.metrics

st.subheader("Métriques")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("Precision", f"{m['precision']:.3f}")
c3.metric("Recall", f"{m['recall']:.3f}")
c4.metric("F1", f"{m['f1']:.3f}")
c5.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")

st.subheader("Matrice de confusion")
cm = art.cm
cm_fig = px.imshow(cm, text_auto=True, aspect="auto",
                    labels=dict(x="Prédit", y="Réel"),
                    x=["0","1"], y=["0","1"],
                    title="Confusion Matrix")
st.plotly_chart(cm_fig, use_container_width=True)

st.subheader("Courbe ROC")
fpr, tpr, _ = art.roc
if fpr.size == 0:
    st.info("ROC non disponible (probabilités indisponibles).")
else:
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aléatoire"))
    roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig, use_container_width=True)

st.subheader("Precision-Recall")
prec, rec, _ = art.pr
if prec.size == 0:
    st.info("PR non disponible (probabilités indisponibles).")
else:
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
    pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(pr_fig, use_container_width=True)
