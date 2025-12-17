import streamlit as st

from utils import load_wine_data, add_target_quality_binary, train_one_model


@st.cache_data(show_spinner=False)
def _get_data(threshold: int):
    df = load_wine_data()
    return add_target_quality_binary(df, threshold=threshold)


def render():
    st.markdown(
        """
        Sélectionne un modèle, entraîne-le **sans fuite de données** :

        - Split Train/Test d'abord (données brutes)
        - `StandardScaler` + `SMOTE` sont dans un pipeline entraîné uniquement sur `X_train`
        """
    )

    model_name = st.selectbox(
        "Modèle",
        [
            "Logistic Regression + SMOTE",
            "k-NN + SMOTE",
            "Decision Tree + SMOTE",
            "Naive Bayes + SMOTE",
            "SVM (RBF) + SMOTE",
        ],
        index=0,
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        threshold = st.slider("Seuil qualité → classe 1 si quality ≥", 5, 8, 7, 1)
    with col_cfg2:
        test_size = st.slider("Taille du test", 0.1, 0.4, 0.2, 0.05)

    df = _get_data(threshold)
    if st.button("🚀 Entraîner le modèle", type="primary"):
        with st.spinner("Entraînement en cours..."):
            pipe, artifacts, splits = train_one_model(
                df,
                model_name=model_name,
                threshold=threshold,
                test_size=test_size,
            )

        st.session_state["last_model_name"] = model_name
        st.session_state["last_pipe"] = pipe
        st.session_state["last_artifacts"] = artifacts
        st.session_state["last_splits_cols"] = splits[0].columns.tolist()

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

        st.caption("Passe à l’onglet Évaluation pour les courbes ROC/PR et la matrice de confusion.")
    else:
        st.info("Clique sur **Entraîner le modèle** pour afficher les métriques.")
