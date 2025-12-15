
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_wine_data, add_target_quality_binary, split_xy

from sklearn.neighbors import LocalOutlierFactor

st.set_page_config(page_title="Préprocessing • Wine Quality", page_icon="🧹", layout="wide")
st.title("🧹 Préprocessing — cible, outliers, fuite de données")

st.markdown(
"""
Ici, on documente la logique du notebook :

- Création de `quality_binary = 1` si `quality >= 7`  
- Détection d'outliers (LOF) **uniquement pour analyse**  
- Règle anti leakage : **Split avant StandardScaler/SMOTE** (fait ensuite via Pipeline)
"""
)

@st.cache_data(show_spinner=False)
def get_data():
    df = load_wine_data()
    df = add_target_quality_binary(df, threshold=7)
    return df

df = get_data()
X, y = split_xy(df)

st.subheader("1) Cible binaire")
fig = px.pie(df, names="quality_binary", title="Répartition de quality_binary")
st.plotly_chart(fig, use_container_width=True)

st.subheader("2) Outliers avec LOF (visualisation)")
cont = st.slider("Contamination (proportion d'outliers)", 0.001, 0.05, 0.02, 0.001)
n_neighbors = st.slider("n_neighbors", 5, 50, 20, 1)

@st.cache_data(show_spinner=False)
def compute_lof(X_in: pd.DataFrame, nn: int, contamination: float):
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    flags = lof.fit_predict(X_in.select_dtypes(include=["int64","float64"]))
    return flags

flags = compute_lof(X, n_neighbors, cont)
df_flags = df.copy()
df_flags["lof_flag"] = flags
df_flags["is_outlier"] = (df_flags["lof_flag"] == -1).astype(int)

c1, c2 = st.columns(2)
c1.metric("Outliers détectés", int(df_flags["is_outlier"].sum()))
c2.metric("Inliers", int((df_flags["is_outlier"] == 0).sum()))

# Simple projection plot: choose two features
num_cols = [c for c in X.columns if c != "wine type"]
xcol = st.selectbox("Feature X", num_cols, index=0)
ycol = st.selectbox("Feature Y", num_cols, index=min(1, len(num_cols)-1))
fig2 = px.scatter(df_flags, x=xcol, y=ycol, color="is_outlier", opacity=0.6, symbol="wine type")
st.plotly_chart(fig2, use_container_width=True)

st.info("Dans le notebook, tu supprimes les outliers pour créer X_clean/y_clean. "
        "Dans une app, tu peux garder cette étape optionnelle (coût/impact à discuter), "
        "mais **l'anti-leakage** reste la priorité : split avant scaler/SMOTE.")
