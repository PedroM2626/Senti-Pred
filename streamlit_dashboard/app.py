import os
import streamlit as st
import json
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")

st.title("Dashboard de Métricas do Modelo Senti-Pred")

st.markdown("""
Este dashboard apresenta as métricas de avaliação e as visualizações geradas pelo pipeline Senti-Pred.
""")

# Paths (relativos à pasta streamlit_dashboard)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METRICS_PATH = os.path.join(BASE, 'reports', 'metrics', 'model_metrics.json')
VIS_DIR = os.path.join(BASE, 'reports', 'visualizacoes')

if not os.path.exists(METRICS_PATH):
    st.error("Arquivo de métricas 'reports/metrics/model_metrics.json' não encontrado. Execute `03_modeling.py` e `04_evaluation.py` para gerar métricas.")
    st.stop()

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

st.header("Resumo")
st.write(f"**Melhor modelo:** {metrics.get('best_model', 'N/A')}")

# Build table of model metrics
rows = []
for name, info in metrics.get('results', {}).items():
    rows.append({
        'model': name,
        'accuracy': info.get('accuracy'),
        'f1_macro': info.get('f1_macro'),
        'roc_auc_macro': info.get('roc_auc_macro'),
        'average_precision_macro': info.get('average_precision_macro'),
        'train_time_s': info.get('train_time_seconds'),
        'predict_time_s': info.get('predict_time_seconds')
    })

if len(rows) > 0:
    df_metrics = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
    st.dataframe(df_metrics)
else:
    st.info('Nenhuma métrica encontrada em results do JSON.')

st.header('Visualizações')
col1, col2 = st.columns(2)
with col1:
    roc_p = os.path.join(VIS_DIR, 'comparison_roc.png')
    if os.path.exists(roc_p):
        st.image(roc_p, caption='ROC Comparativo', use_column_width=True)
    else:
        st.warning('ROC comparativo não encontrado (reports/visualizacoes/comparison_roc.png)')
with col2:
    pr_p = os.path.join(VIS_DIR, 'comparison_pr.png')
    if os.path.exists(pr_p):
        st.image(pr_p, caption='PR Comparativo', use_column_width=True)
    else:
        st.warning('PR comparativo não encontrado (reports/visualizacoes/comparison_pr.png)')

cm_p = os.path.join(VIS_DIR, 'comparison_confusion_matrices.png')
if os.path.exists(cm_p):
    st.header('Matrizes de Confusão Comparativas')
    st.image(cm_p, use_column_width=True)
else:
    st.warning('Matrizes de confusão não encontradas (reports/visualizacoes/comparison_confusion_matrices.png)')

st.header('Relatório completo (JSON)')
st.json(metrics)