
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Interprétabilité • Wine Quality", page_icon="🔍", layout="wide")
st.title("🔍 Interprétabilité — importance / coefficients (selon modèle)")

if "last_pipe" not in st.session_state:
    st.warning("Aucun modèle entraîné. Va d’abord dans **Modélisation** puis entraîne un modèle.")
    st.stop()

pipe = st.session_state["last_pipe"]
model_name = st.session_state.get("last_model_name", "Modèle")
st.subheader(f"Modèle : {model_name}")

# Try to extract feature names (numeric + wine type is categorical in raw dataset)
# For simplicity, we assume 'wine type' was kept as string and imputed/scaled will ignore non-numeric.
# In a production version, you'd encode 'wine type' via OneHotEncoder in a ColumnTransformer.

# Determine feature columns from last training split if available
if "last_splits_cols" in st.session_state:
    cols = st.session_state["last_splits_cols"]
else:
    cols = None

model = pipe.named_steps.get("model", None)

if model is None:
    st.info("Impossible d'extraire le modèle du pipeline.")
    st.stop()

# Logistic regression coefficients
if hasattr(model, "coef_"):
    coef = model.coef_.ravel()
    if cols is None:
        cols = [f"feature_{i}" for i in range(len(coef))]
    df_coef = pd.DataFrame({"feature": cols[:len(coef)], "coef": coef}).sort_values("coef", ascending=False)
    fig = px.bar(df_coef.head(20), x="feature", y="coef", title="Top coefficients (logreg)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_coef, use_container_width=True)

# Tree feature importance
elif hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    if cols is None:
        cols = [f"feature_{i}" for i in range(len(importances))]
    df_imp = pd.DataFrame({"feature": cols[:len(importances)], "importance": importances}).sort_values("importance", ascending=False)
    fig = px.bar(df_imp.head(20), x="feature", y="importance", title="Top feature importances (tree)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_imp, use_container_width=True)

else:
    st.info("Ce modèle n'expose pas facilement des coefficients / importances (ex: SVM RBF). "
            "Pour une interprétabilité avancée : SHAP (hors scope minimal).")
