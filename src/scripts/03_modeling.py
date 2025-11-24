# Modelagem - Senti-Pred
# Este notebook contém o desenvolvimento e treinamento dos modelos de análise de sentimentos para o projeto Senti-Pred.

# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys
from pathlib import Path

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
# %matplotlib inline (Este comando é específico de IPython/Jupyter e será removido ou comentado)

# Carregar os dados processados (caminho relativo ao repositório)
project_root = Path(__file__).resolve().parents[2]
processed_path = project_root / 'data' / 'processed' / 'processed_data.csv'
if not processed_path.exists():
    raise FileNotFoundError(f"Arquivo de dados processados não encontrado: {processed_path}")
df = pd.read_csv(processed_path)

# Exibir as primeiras linhas
df.head()

# Preparar dados para modelagem
if 'text_lemmatized' in df.columns and 'sentiment' in df.columns:
    X = df['text_lemmatized']
    y = df['sentiment']
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

## Modelo 1: Regressão Logística com TF-IDF

# Pipeline para Regressão Logística
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

# Treinar o modelo
lr_pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred_lr = lr_pipeline.predict(X_test)
print("Relatório de Classificação - Regressão Logística:")
print(classification_report(y_test, y_pred_lr))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lr_pipeline.classes_, yticklabels=lr_pipeline.classes_)
plt.title('Matriz de Confusão - Regressão Logística')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

## Modelo 2: Naive Bayes Multinomial

# Pipeline para Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# Treinar o modelo
nb_pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred_nb = nb_pipeline.predict(X_test)
print("Relatório de Classificação - Naive Bayes:")
print(classification_report(y_test, y_pred_nb))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=nb_pipeline.classes_, yticklabels=nb_pipeline.classes_)
plt.title('Matriz de Confusão - Naive Bayes')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

## Modelo 3: Random Forest

# Pipeline para Random Forest
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treinar o modelo
rf_pipeline.fit(X_train, y_train)

# Avaliar o modelo
y_pred_rf = rf_pipeline.predict(X_test)
print("Relatório de Classificação - Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=rf_pipeline.classes_, yticklabels=rf_pipeline.classes_)
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

## Comparação dos Modelos

# Comparar acurácia dos modelos
models = {
    'Regressão Logística': (lr_pipeline, y_pred_lr),
    'Naive Bayes': (nb_pipeline, y_pred_nb),
    'Random Forest': (rf_pipeline, y_pred_rf)
}

accuracies = {}
for name, (model, y_pred) in models.items():
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name}: {acc:.4f}")

# Visualizar comparação
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'orange'])
plt.title('Comparação de Acurácia entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.show()

## Salvar o Melhor Modelo

# Identificar o melhor modelo
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name][0]
print(f"Melhor modelo: {best_model_name} com acurácia de {accuracies[best_model_name]:.4f}")

# Salvar o modelo (em `src/models` do repositório)
model_path = project_root / 'src' / 'models' / 'sentiment_model.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, str(model_path))
print(f"Modelo salvo em: {model_path}")

## Conclusões da Modelagem
# - Resumo dos modelos testados
# - Análise do desempenho do melhor modelo
# - Próximos passos para avaliação e implantação