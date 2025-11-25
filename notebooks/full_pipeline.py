
# -*- coding: utf-8 -*-
"""full_pipeline.py

SENTI-PRED: Pipeline completo (EDA, pré-processamento em inglês, modelagem
treinada em `twitter_training.csv` e avaliada em `twitter_validation.csv`).

Fluxo principal:
- Carrega `data/raw/twitter_training.csv` e `data/raw/twitter_validation.csv`
- EDA básica
- Pré-processamento (inglês): limpeza, remoção de stopwords, lematização com POS
- Treina múltiplos classificadores (LogisticRegression, MultinomialNB, LinearSVC)
- Avalia modelos na base de validação e salva o melhor em `src/models/sentiment_model.pkl`
- Salva métricas em JSON e gráficos em `reports/visualizacoes/`
"""

import os
import re
import json
import warnings
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from IPython.display import display

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

print("[START] full_pipeline.py — EDA, Preprocess (EN), Modeling")

# `BASE_DIR` aponta para a raiz do repositório (pai da pasta `notebooks`)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'twitter_training.csv')
VAL_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'twitter_validation.csv')
TRAIN_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'processed_train.csv')
VAL_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'processed_validation.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src', 'models', 'sentiment_model.pkl')
VIS_DIR = os.path.join(BASE_DIR, 'reports', 'visualizacoes')
METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')

os.makedirs(os.path.dirname(TRAIN_PROCESSED), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# -----------------------------
# NLTK resources (English)
# -----------------------------
print('[INFO] Baixando recursos NLTK (se necessário)...')
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
for r in nltk_resources:
    try:
        nltk.download(r, quiet=True)
    except Exception:
        pass
print('[OK] Recursos NLTK prontos')

# -----------------------------
# Carregamento de dados
# -----------------------------
if not os.path.exists(TRAIN_RAW) or not os.path.exists(VAL_RAW):
    raise FileNotFoundError(f"Esperado arquivos em '{TRAIN_RAW}' e '{VAL_RAW}' — coloque os CSVs em data/raw/")

cols = ['tweet_id', 'entity', 'sentiment', 'text']
print(f"[INFO] Carregando '{TRAIN_RAW}' e '{VAL_RAW}' (assumindo sem cabeçalho)...")
df_train = pd.read_csv(TRAIN_RAW, names=cols, header=None, engine='python', encoding='utf-8')
df_val = pd.read_csv(VAL_RAW, names=cols, header=None, engine='python', encoding='utf-8')

df_train['split'] = 'train'
df_val['split'] = 'validation'
df = pd.concat([df_train, df_val], ignore_index=True)
print(f"[OK] Dados carregados: train={len(df_train)} | validation={len(df_val)} | total={len(df)}")

# -----------------------------
# EDA rápida
# -----------------------------
print('\n[EDA] Primeiras linhas (combined)')
display(df.head())
print('\n[EDA] Informações gerais')
display(df.info())

if 'sentiment' in df.columns:
    print('\n[EDA] Distribuição de sentimentos (combined)')
    display(df['sentiment'].value_counts())

# text length
text_col = 'text'
df['text_length'] = df[text_col].astype(str).apply(lambda s: len(s.split()))
plt.figure(figsize=(10, 5))
sns.histplot(df['text_length'], bins=40, kde=True)
plt.title('Distribuição de comprimento de texto')
plt.xlabel('Número de palavras')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'text_length.png'))
plt.close()

# top words (raw) — useful before preprocessing
all_words = ' '.join(df[text_col].astype(str)).lower().split()
top_raw = pd.Series(all_words).value_counts().head(20)
plt.figure(figsize=(12, 5))
top_raw.plot(kind='bar')
plt.title('Top words (raw)')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'top_words_raw.png'))
plt.close()

print('[OK] EDA concluída')

# -----------------------------
# Pré-processamento (inglês)
# -----------------------------
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_en(text):
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text, language='english')
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered)

def lemmatize_text_en(text):
    if not isinstance(text, str):
        return ''
    tokens = word_tokenize(text, language='english')
    try:
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        pos_tags = [(t, '') for t in tokens]

    def _get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    lemmas = []
    for token, tag in pos_tags:
        wn_tag = _get_wordnet_pos(tag) if tag else wordnet.NOUN
        lemmas.append(lemmatizer.lemmatize(token, wn_tag))
    return ' '.join(lemmas)

def preprocess_series(series):
    s = series.fillna('').astype(str)
    s_clean = s.apply(clean_text)
    s_nostop = s_clean.apply(remove_stopwords_en)
    s_lemma = s_nostop.apply(lemmatize_text_en)
    return s_lemma

print('[INFO] Aplicando pré-processamento em train e validation (inglês)')
df_train_proc = df_train.copy()
df_val_proc = df_val.copy()

df_train_proc['text_clean'] = df_train_proc['text'].apply(clean_text)
df_train_proc['text_no_stop'] = df_train_proc['text_clean'].apply(remove_stopwords_en)
df_train_proc['text_lemmatized'] = df_train_proc['text_no_stop'].apply(lemmatize_text_en)

df_val_proc['text_clean'] = df_val_proc['text'].apply(clean_text)
df_val_proc['text_no_stop'] = df_val_proc['text_clean'].apply(remove_stopwords_en)
df_val_proc['text_lemmatized'] = df_val_proc['text_no_stop'].apply(lemmatize_text_en)

print(f"[OK] Pré-processamento concluído (dados mantidos em memória)")

# -----------------------------
# Modelagem: treinar com train -> predizer validation
# -----------------------------
print('\n[MODELING] Treinando modelos e avaliando na base de validação')

X_train = df_train_proc['text_lemmatized'].astype(str)
y_train = df_train_proc['sentiment']
X_val = df_val_proc['text_lemmatized'].astype(str)
y_val = df_val_proc['sentiment']

# Remover vazios
mask_train = X_train.str.strip().replace('', np.nan).notna()
mask_val = X_val.str.strip().replace('', np.nan).notna()
X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_val = X_val[mask_val]
y_val = y_val[mask_val]

print(f"Treino: {len(X_train)} | Validation: {len(X_val)}")

models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
    'MultinomialNB': MultinomialNB(),
    'LinearSVC': LinearSVC(max_iter=20000, random_state=42)
}

results = {}
for name, clf in models.items():
    print(f"\n[MODEL] Treinando {name}...")
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2))),
        ('clf', clf)
    ])
    # Medir tempo de treinamento
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t1 = time.time()
    train_time = t1 - t0

    # Medir tempo de predição (opcional)
    t0p = time.time()
    preds = pipe.predict(X_val)
    t1p = time.time()
    predict_time = t1p - t0p

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds)

    # Prepare scores for ROC/PR (one-vs-rest). Binarize labels
    classes = np.unique(y_val)
    y_val_b = label_binarize(y_val, classes=classes)

    # Try to obtain scores: prefer predict_proba, fall back to decision_function
    y_score = None
    try:
        y_score = pipe.predict_proba(X_val)
    except Exception:
        try:
            decision = pipe.decision_function(X_val)
            # If binary, make it 2D
            if decision.ndim == 1:
                decision = np.vstack([-decision, decision]).T
            y_score = decision
        except Exception:
            y_score = None

    roc_auc_macro = None
    avg_precision_macro = None
    # Compute ROC AUC and Average Precision (macro) when scores available
    if y_score is not None and y_score.shape[1] == y_val_b.shape[1]:
        try:
            roc_auc_macro = roc_auc_score(y_val_b, y_score, average='macro', multi_class='ovr')
        except Exception:
            roc_auc_macro = None
        try:
            # average_precision_score supports multilabel indicator
            avg_precision_macro = average_precision_score(y_val_b, y_score, average='macro')
        except Exception:
            avg_precision_macro = None

    results[name] = {
        'pipeline': pipe,
        'accuracy': acc,
        'f1_macro': f1,
        'roc_auc_macro': roc_auc_macro,
        'average_precision_macro': avg_precision_macro,
        'train_time_seconds': train_time,
        'predict_time_seconds': predict_time,
        'report': report,
            'confusion_matrix': cm.tolist(),
            'y_score': y_score
    }

    print(f"[RESULT] {name} — Accuracy: {acc:.4f} | F1-macro: {f1:.4f} | ROC-AUC(macro): {str(roc_auc_macro)} | AP(macro): {str(avg_precision_macro)}")
    print(classification_report(y_val, preds))
        # não salvar plots individuais por modelo aqui; plots comparativos serão gerados após o loop

# Escolher melhor por F1-macro
best = max(results.keys(), key=lambda k: results[k]['f1_macro'])
best_pipeline = results[best]['pipeline']
print(f"\n[OK] Melhor modelo: {best} (F1-macro={results[best]['f1_macro']:.4f})")

# Salvar modelo
joblib.dump(best_pipeline, MODEL_PATH)
print(f"[OK] Modelo salvo em: {MODEL_PATH}")

# Salvar métricas detalhadas por modelo
metrics_out = {
    'best_model': best,
    'results': {}
}
for k in results:
    metrics_out['results'][k] = {
        'accuracy': results[k]['accuracy'],
        'f1_macro': results[k]['f1_macro'],
        'roc_auc_macro': results[k].get('roc_auc_macro'),
        'average_precision_macro': results[k].get('average_precision_macro'),
        'train_time_seconds': results[k].get('train_time_seconds'),
        'predict_time_seconds': results[k].get('predict_time_seconds'),
        'classification_report': results[k]['report'],
        'confusion_matrix': results[k]['confusion_matrix']
    }

with open(os.path.join(METRICS_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"[OK] Métricas salvas em: {os.path.join(METRICS_DIR, 'model_metrics.json')}")

# Gerar gráficos comparativos (um único ROC com todas as modelos, um único PR, e
# uma figura com as matrizes de confusão lado a lado). Não geramos mais CSV resumo.
print('\n[COMPARISON PLOTS] Gerando gráficos comparativos: ROC, PR e Confusion matrices')

# classes e etiqueta binarizada (mesma ordem para todos os modelos)
classes_all = np.unique(y_val)
y_val_b_all = label_binarize(y_val, classes=classes_all)

# --- ROC comparativo (micro/macro averaged via ravel) ---
plt.figure(figsize=(8, 6))
plotted_any = False
for name in results:
    y_score = results[name].get('y_score')
    if y_score is None:
        continue
    try:
        # micro-averaged curve across classes (ravel)
        fpr, tpr, _ = roc_curve(y_val_b_all.ravel(), y_score.ravel())
        auc_val = None
        try:
            auc_val = roc_auc_score(y_val_b_all, y_score, average='macro', multi_class='ovr')
        except Exception:
            auc_val = None
        label = f"{name}"
        if auc_val is not None:
            label += f" (AUC={auc_val:.3f})"
        plt.plot(fpr, tpr, lw=2, label=label)
        plotted_any = True
    except Exception:
        continue
if plotted_any:
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curves (all models)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(VIS_DIR, 'comparison_roc.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"[OK] ROC comparativo salvo em: {roc_path}")
else:
    print('[WARN] Nenhum score disponível para plotting ROC comparativo')

# --- Precision-Recall comparativo ---
plt.figure(figsize=(8, 6))
plotted_any = False
for name in results:
    y_score = results[name].get('y_score')
    if y_score is None:
        continue
    try:
        precision, recall, _ = precision_recall_curve(y_val_b_all.ravel(), y_score.ravel())
        ap = None
        try:
            ap = average_precision_score(y_val_b_all, y_score, average='macro')
        except Exception:
            ap = None
        label = f"{name}"
        if ap is not None:
            label += f" (AP={ap:.3f})"
        plt.plot(recall, precision, lw=2, label=label)
        plotted_any = True
    except Exception:
        continue
if plotted_any:
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comparative Precision-Recall Curves (all models)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    pr_path = os.path.join(VIS_DIR, 'comparison_pr.png')
    plt.savefig(pr_path)
    plt.close()
    print(f"[OK] Precision-Recall comparativo salvo em: {pr_path}")
else:
    print('[WARN] Nenhum score disponível para plotting Precision-Recall comparativo')

# --- Confusion matrices lado a lado ---
model_names = list(results.keys())
n_models = len(model_names)
cms = [np.array(results[nm]['confusion_matrix']) for nm in model_names]
if len(cms) > 0:
    vmax = max(cm.max() for cm in cms)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, nm, cm in zip(axes, model_names, cms):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes_all, yticklabels=classes_all,
                    vmin=0, vmax=vmax, ax=ax)
        ax.set_title(f'Confusion — {nm}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
    plt.tight_layout()
    cm_path = os.path.join(VIS_DIR, 'comparison_confusion_matrices.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Matrizes de confusão comparativas salvas em: {cm_path}")
else:
    print('[WARN] Nenhuma matriz de confusão disponível para plotagem')

# -----------------------------
# Análises finais (distribuições e top-words pós-processamento)
# -----------------------------
print('\n[ANALYSIS] Gerando distribuições e top-words')
dist = pd.concat([df_train_proc.assign(split='train'), df_val_proc.assign(split='validation')]).groupby(['split', 'sentiment']).size().unstack(fill_value=0)
plt.figure(figsize=(8, 5))
dist.plot(kind='bar', stacked=True)
plt.title('Distribuição de Sentimentos por Split')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'sentiment_distribution_by_split.png'))
plt.close()

def top_words(series, n=20):
    allw = ' '.join(series.fillna('').astype(str)).lower().split()
    return pd.Series(allw).value_counts().head(n)

top_overall = top_words(pd.concat([df_train_proc['text_lemmatized'], df_val_proc['text_lemmatized']]))
plt.figure(figsize=(12, 5))
top_overall.plot(kind='bar')
plt.title('Top words (processed)')
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'top_words_overall.png'))
plt.close()

for split_name, dframe in [('train', df_train_proc), ('validation', df_val_proc)]:
    tw = top_words(dframe['text_lemmatized'])
    plt.figure(figsize=(12, 5))
    tw.plot(kind='bar')
    plt.title(f'Top words ({split_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f'top_words_{split_name}.png'))
    plt.close()

print('[OK] Pipeline finalizado')