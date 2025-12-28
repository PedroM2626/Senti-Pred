import os
import streamlit as st
import json
import pandas as pd
from PIL import Image
import numpy as np
import joblib

st.set_page_config(layout="wide")

st.title("Dashboard de Métricas do Modelo Senti-Pred")

st.markdown("""
Este dashboard apresenta as métricas de avaliação e as visualizações geradas pelo pipeline Senti-Pred.
""")

# Paths (relativos à pasta streamlit_dashboard)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METRICS_PATH = os.path.join(BASE, 'reports', 'metrics', 'model_metrics.json')
VIS_DIR = os.path.join(BASE, 'reports', 'visualizacoes')
MODEL_PATH = os.path.join(BASE, 'src', 'models', 'sentiment_model.pkl')

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

# Predição com o modelo (texto único)
st.header('Predição com o Modelo')
if not os.path.exists(MODEL_PATH):
    st.error("Modelo não encontrado em 'src/models/sentiment_model.pkl'. Execute `03_modeling.py` para treinar e salvar o modelo.")
else:
    @st.cache_resource
    def _load_model(path: str):
        return joblib.load(path)

    model = _load_model(MODEL_PATH)

    texto = st.text_area("Digite um texto para análise de sentimento", "")
    if st.button("Prever sentimento"):
        t = (texto or "").strip()
        if not t:
            st.warning("Digite um texto válido para realizar a predição.")
        else:
            try:
                pred = model.predict([t])[0]
                conf = None
                try:
                    scores = model.decision_function([t])
                    s = np.array(scores)[0]
                    exps = np.exp(s - np.max(s))
                    probs = exps / exps.sum()
                    conf = float(probs.max())
                except Exception:
                    pass

                label = str(pred)
                if label.lower().startswith("pos"):
                    st.success(f"Sentimento: {label}" + (f" — confiança≈{conf:.2f}" if conf is not None else ""))
                elif label.lower().startswith("neg"):
                    st.error(f"Sentimento: {label}" + (f" — confiança≈{conf:.2f}" if conf is not None else ""))
                else:
                    st.info(f"Sentimento: {label}" + (f" — confiança≈{conf:.2f}" if conf is not None else ""))
            except Exception as e:
                st.exception(e)

# Predição em lote (upload de CSV)
st.header('Predição em Lote (CSV)')
if not os.path.exists(MODEL_PATH):
    st.info("Para usar predições em lote, gere o modelo primeiro com `03_modeling.py`.")
else:
    upload = st.file_uploader("Envie um CSV com a coluna 'text'", type=['csv'])
    if upload is not None:
        try:
            df_in = pd.read_csv(upload)
        except Exception as e:
            st.error(f"Não foi possível ler o CSV: {e}")
            df_in = None

        if df_in is not None:
            if 'text' not in df_in.columns:
                st.error("CSV deve conter a coluna 'text'.")
            else:
                model = _load_model(MODEL_PATH)
                texts = df_in['text'].astype(str).fillna("")
                try:
                    preds = model.predict(texts.tolist())
                except Exception as e:
                    st.exception(e)
                    preds = None

                if preds is not None:
                    df_out = df_in.copy()
                    df_out['predicted_sentiment'] = preds
                    st.subheader("Amostra das predições")
                    st.dataframe(df_out.head(300))

                    st.subheader("Distribuição de sentimentos previstos")
                    counts = pd.Series(preds).value_counts().reset_index()
                    counts.columns = ['sentiment', 'count']
                    st.bar_chart(counts.set_index('sentiment'))

                    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar resultados (CSV)",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
