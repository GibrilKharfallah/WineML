import streamlit as st
import plotly.express as px

from utils import load_wine_data, add_target_quality_binary


@st.cache_data(show_spinner=False)
def _get_data():
    df = load_wine_data()
    return add_target_quality_binary(df, threshold=7)


def render():
    try:
        df = _get_data()
    except Exception as exc:
        st.error("Impossible de charger les données. Vérifie les chemins ou uploade les CSV.")
        st.exception(exc)
        return

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

    st.info(
        "Utilise les onglets ci-dessus (EDA, Préprocessing, Modélisation, "
        "Évaluation, Interprétabilité) pour suivre le flux complet du projet."
    )
