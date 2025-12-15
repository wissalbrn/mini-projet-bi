# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="Mini-Projet BI",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Mini-Projet BI – Analyse & Modélisation")
st.markdown("**Application web interactive professionnelle**")


st.sidebar.header("📂 Données")
dataset_choice = st.sidebar.selectbox(
    "Choisir un dataset",
    ["Heart Disease", "Breast Cancer", "Diabetes (Pima)", "Upload CSV"]
)


if dataset_choice == "Heart Disease":
    df = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")
elif dataset_choice == "Breast Cancer":
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
elif dataset_choice == "Diabetes (Pima)":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        names=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","target"]
    )
else:
    uploaded = st.sidebar.file_uploader("Uploader un fichier CSV", type="csv")
    if uploaded is None:
        st.warning("Veuillez uploader un fichier CSV")
        st.stop()
    df = pd.read_csv(uploaded)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Données",
    "🛠 Prétraitement",
    "🤖 Modélisation",
    "📈 Résultats",
    "🔮 Prédiction"
])


with tab1:
    st.subheader("Aperçu du dataset")
    st.success(f"{df.shape[0]} lignes × {df.shape[1]} colonnes")
    st.dataframe(df.head())
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    st.subheader("Valeurs manquantes")
    st.write(df.isna().sum())


with tab2:
    if "target" not in df.columns:
        st.error("❌ Le dataset doit contenir une colonne 'target'")
        st.stop()

    
    X = df.drop("target", axis=1)
    y = df["target"]

    
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)

        
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    
    df_clean = pd.concat([X, y], axis=1).drop_duplicates()
    X = df_clean.drop("target", axis=1)
    y = df_clean["target"]

    
    if dataset_choice == "Heart Disease":
        X['chol_age_ratio'] = X['chol'] / X['age']
        X['thalach_age_ratio'] = X['thalach'] / X['age']
        X['oldpeak_binary'] = (X['oldpeak'] > 1).astype(int)
        num_cols += ['chol_age_ratio', 'thalach_age_ratio', 'oldpeak_binary']

    st.success(" Nettoyage et feature engineering terminés")

    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])
    X_processed = preprocessor.fit_transform(X)

    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_processed, y)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    st.success("✅ Prétraitement + SMOTE + Train/Test split terminé")


with tab3:
    model_choice = st.selectbox(
        "Choisir un modèle",
        ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Gradient Boosting"]
    )

    if model_choice == "Random Forest":
        n_estimators = st.slider("Nombre d’arbres", 50, 300, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "SVM":
        C = st.slider("Paramètre C", 0.01, 10.0, 1.0)
        model = SVC(C=C, probability=True)
    elif model_choice == "Gradient Boosting":
        n_estimators = st.slider("Nombre d’arbres", 50, 300, 100)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    elif model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    st.success("✅ Modèle entraîné avec succès")


with tab4:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    st.subheader("📊 Métriques de performance")
    st.table(pd.DataFrame(metrics, index=["Valeur"]).T)

    
    col1, col2, col3 = st.columns(3)

    
    num_cols_corr = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols_corr].corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        text_auto=True,
        title="Matrice de corrélation"
    )
    col1.plotly_chart(fig_corr, use_container_width=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC Curve", line=dict(color='blue')))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", line=dict(dash="dash", color='red')))
    fig_roc.update_layout(title="Courbe ROC", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    col2.plotly_chart(fig_roc, use_container_width=True)

    #  Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Matrice de Confusion")
    col3.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")  

   
    col4, col5 = st.columns(2)

    # Histogramme de la variable target
    fig_target = px.histogram(df, x="target", color="target", text_auto=True, title="Distribution de la variable cible")
    col4.plotly_chart(fig_target, use_container_width=True)

    # Feature importance pour RF, GB, XGBoost
    if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        fig_imp = px.bar(feat_imp.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Features Importance")
        col5.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    
    export = st.selectbox("Exporter métriques", ["CSV", "JSON"])
    if st.button("Exporter"):
        df_m = pd.DataFrame(metrics, index=[0])
        if export == "CSV":
            df_m.to_csv("metrics.csv", index=False)
        else:
            df_m.to_json("metrics.json", orient="records")
        st.success("Export réussi")


with tab5:
    st.subheader("🔮 Nouvelle prédiction")
    inputs = []
    for col in X.columns:
        val = st.number_input(col, float(X[col].min()), float(X[col].max()))
        inputs.append(val)

    if st.button("Prédire"):
        X_new = preprocessor.transform([inputs])
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0][1]
        st.success(f"Classe prédite : {pred} | Probabilité : {proba:.2f}")

