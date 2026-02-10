import streamlit as st
from views import dashboard, preprocessing, modeling, evaluation, interpretability

st.set_page_config(
    page_title="Wine Quality • Dashboard ML",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Dashboard Machine Learning — Wine Quality")
st.caption("Projet ML : exploration, préprocessing, modélisation, évaluation et interprétabilité.")

with st.expander("📌 Contexte et objectif", expanded=True):
    st.markdown(
        """
        Ce projet explore la qualité des vins rouges et blancs à partir de mesures physico-chimiques.
        L’objectif est de construire un modèle de classification simple et robuste, puis d’expliquer
        ses résultats via des métriques, des courbes d’évaluation et une interprétabilité des features.
        """
    )

tab_dash, tab_prep, tab_model, tab_eval, tab_interp = st.tabs(
    ["📊 EDA", "🧹 Préprocessing", "🤖 Modélisation", "🧪 Évaluation", "🔍 Interprétabilité"]
)

with tab_dash:
    dashboard.render()

with tab_prep:
    preprocessing.render()

with tab_model:
    modeling.render()

with tab_eval:
    evaluation.render()

with tab_interp:
    interpretability.render()
