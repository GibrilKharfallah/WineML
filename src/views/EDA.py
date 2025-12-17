
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_wine_data, add_target_quality_binary

st.set_page_config(page_title="EDA • Wine Quality", page_icon="📊", layout="wide")
st.title("📊 EDA — Exploration des données")

@st.cache_data(show_spinner=False)
def get_data():
    df = load_wine_data()
    df = add_target_quality_binary(df, threshold=7)
    return df

df = get_data()

st.markdown("### 1) Statistiques descriptives")
st.dataframe(df.describe(include="all").T, use_container_width=True)

st.markdown("### 2) Corrélations (numérique)")
num_df = df.select_dtypes(include=["int64", "float64"])
corr = num_df.corr(numeric_only=True)
fig_corr = px.imshow(corr, aspect="auto", title="Matrice de corrélation (variables numériques)")
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("### 3) Distributions par variable")
num_cols = [c for c in num_df.columns if c not in ["quality", "quality_binary"]]
col = st.selectbox("Variable", num_cols, index=0)
fig_hist = px.histogram(df, x=col, color="quality_binary", marginal="box", nbins=50)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### 4) Comparaison Rouge vs Blanc")
xcol = st.selectbox("X", num_cols, index=0, key="xcol")
ycol = st.selectbox("Y", num_cols, index=min(1, len(num_cols)-1), key="ycol")
fig_scatter = px.scatter(df, x=xcol, y=ycol, color="wine type", opacity=0.6, trendline="ols")
st.plotly_chart(fig_scatter, use_container_width=True)
