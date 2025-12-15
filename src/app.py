import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_wine_data, add_target_quality_binary

st.set_page_config(
    page_title="Wine Quality • Dashboard ML",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Dashboard Machine Learning — Wine Quality")
st.caption("Projet ML : exploration, préprocessing, modélisation (SMOTE + pipelines), évaluation et interprétabilité.")

with st.expander("📌 Contexte et objectif", expanded=True):
    st.markdown(
        """
        **Données :** UCI Wine Quality (rouge & blanc), variables physico-chimiques + `wine type`.

        **Objectif :** prédire une **qualité binaire** :
        - `1` si `quality >= 7` (vin “bon”)
        - `0` sinon

        **Important (anti data leakage) :**
        - Split Train/Test **avant** tout `StandardScaler` / `SMOTE`
        - Standardisation et SMOTE sont dans un **Pipeline** entraîné uniquement sur le train.
        """
    )

@st.cache_data(show_spinner=False)
def get_data():
    df = load_wine_data()
    df = add_target_quality_binary(df, threshold=7)
    return df

try:
    df = get_data()
except Exception as e:
    st.error("Impossible de charger les données. Vérifie les chemins par défaut ou uploade les CSV.")
    st.exception(e)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{df.shape[0]:,}".replace(",", " "))
c2.metric("Colonnes", df.shape[1])
c3.metric("Vins rouges", int((df["wine type"] == "red").sum()))
c4.metric("Vins blancs", int((df["wine type"] == "white").sum()))

st.subheader("Aperçu")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Distribution de la cible")
fig = px.histogram(df, x="quality_binary", color="wine type", barmode="group")
st.plotly_chart(fig, use_container_width=True)

st.info("Utilise le menu de gauche (Pages) pour naviguer : EDA, Préprocessing, Modélisation, Évaluation, Interprétabilité.")
