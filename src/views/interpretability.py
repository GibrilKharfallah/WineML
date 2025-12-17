import streamlit as st
import pandas as pd
import plotly.express as px


def render():
    if "last_pipe" not in st.session_state:
        st.warning("Aucun modèle entraîné. Va d’abord dans l’onglet **Modélisation**.")
        return

    pipe = st.session_state["last_pipe"]
    model_name = st.session_state.get("last_model_name", "Modèle")
    st.subheader(f"Modèle : {model_name}")

    cols = st.session_state.get("last_splits_cols")
    model = pipe.named_steps.get("model", None)

    if model is None:
        st.info("Impossible d'extraire le modèle du pipeline.")
        return

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        if cols is None:
            cols = [f"feature_{i}" for i in range(len(coef))]
        df_coef = pd.DataFrame({"feature": cols[: len(coef)], "coef": coef}).sort_values(
            "coef", ascending=False
        )
        fig = px.bar(df_coef.head(20), x="feature", y="coef", title="Top coefficients (logreg)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_coef, use_container_width=True)

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if cols is None:
            cols = [f"feature_{i}" for i in range(len(importances))]
        df_imp = pd.DataFrame({"feature": cols[: len(importances)], "importance": importances}).sort_values(
            "importance", ascending=False
        )
        fig = px.bar(df_imp.head(20), x="feature", y="importance", title="Top feature importances (tree)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_imp, use_container_width=True)
    else:
        st.info(
            "Ce modèle n'expose pas facilement des coefficients / importances (ex: SVM RBF). "
            "Pour aller plus loin : SHAP, LIME, etc."
        )
