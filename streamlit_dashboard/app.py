import streamlit as st
import json
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(layout="wide")

st.title("Dashboard de Métricas do Modelo Senti-Pred")

st.markdown("""
Este dashboard apresenta as métricas de avaliação do modelo de análise de sentimentos Senti-Pred.
""")

# Carregar métricas do modelo
try:
    with open('../reports/metrics/model_metrics.json', 'r') as f:
        metrics = json.load(f)
except FileNotFoundError:
    st.error("Arquivo de métricas 'model_metrics.json' não encontrado. Por favor, execute o script de avaliação primeiro.")
    st.stop()

st.header("Métricas de Classificação")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Acurácia", f"{metrics['accuracy']:.4f}")
with col2:
    st.metric("Precisão (Macro Avg)", f"{metrics['classification_report']['macro avg']['precision']:.4f}")
with col3:
    st.metric("Recall (Macro Avg)", f"{metrics['classification_report']['macro avg']['recall']:.4f}")
with col4:
    st.metric("F1-Score (Macro Avg)", f"{metrics['classification_report']['macro avg']['f1-score']:.4f}")

st.subheader("Relatório de Classificação Detalhado")
st.json(metrics['classification_report'])

st.header("Matriz de Confusão")
try:
    confusion_matrix_image = Image.open('../reports/metrics/confusion_matrix.png')
    st.image(confusion_matrix_image, caption='Matriz de Confusão', use_container_width=True)
except FileNotFoundError:
    st.warning("Imagem da Matriz de Confusão não encontrada.")

st.header("Curva ROC")
try:
    roc_curve_image = Image.open('../reports/metrics/roc_pr_curve.png')
    st.image(roc_curve_image, caption='Curva ROC', use_container_width=True)
except FileNotFoundError:
    st.warning("Imagem da Curva ROC não encontrada.")

st.header("Curva Precision-Recall")
try:
    pr_curve_image = Image.open('../reports/metrics/roc_pr_curve.png')
    st.image(pr_curve_image, caption='Curva Precision-Recall', use_container_width=True)
except FileNotFoundError:
    st.warning("Imagem da Curva Precision-Recall não encontrada.")