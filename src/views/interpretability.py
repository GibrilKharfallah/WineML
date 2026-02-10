import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def _resolve_feature_names(pipe, cols, n_features: int) -> list[str]:
    feature_names = None
    preprocessor = pipe.named_steps.get("preprocess") if pipe is not None else None
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            if cols is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out(cols)
                except Exception:
                    feature_names = None

    if feature_names is None:
        if cols is not None:
            feature_names = cols
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]

    return list(feature_names)


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
        coef = model.coef_
        title = "Top coefficients (logreg)"
        if coef.ndim == 2:
            if coef.shape[0] == 1:
                coef = coef[0]
            else:
                coef = np.mean(np.abs(coef), axis=0)
                title = "Top coefficients (mean |coef|)"
        coef = np.ravel(coef)

        feature_names = _resolve_feature_names(pipe, cols, n_features=len(coef))
        if len(feature_names) != len(coef):
            min_len = min(len(feature_names), len(coef))
            feature_names = feature_names[:min_len]
            coef = coef[:min_len]
            st.caption("Ajustement des longueurs features/coefficients pour affichage.")

        df_coef = pd.DataFrame({"feature": feature_names, "coef": coef}).sort_values(
            "coef", ascending=False
        )
        fig = px.bar(df_coef.head(20), x="feature", y="coef", title=title)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df_coef, width="stretch")

    elif hasattr(model, "feature_importances_"):
        importances = np.ravel(model.feature_importances_)
        feature_names = _resolve_feature_names(pipe, cols, n_features=len(importances))
        if len(feature_names) != len(importances):
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]
            st.caption("Ajustement des longueurs features/importances pour affichage.")

        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        fig = px.bar(df_imp.head(20), x="feature", y="importance", title="Top feature importances (tree)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df_imp, width="stretch")
    else:
        st.info(
            "Ce modèle n'expose pas facilement des coefficients / importances (ex: SVM RBF). "
            "Pour aller plus loin : SHAP, LIME, etc."
        )
