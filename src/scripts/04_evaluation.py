# Avaliação do Modelo - Senti-Pred
# Este notebook contém a avaliação detalhada do modelo de análise de sentimentos para o projeto Senti-Pred.

# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import joblib
import os
import sys
from sklearn.model_selection import train_test_split

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
# %matplotlib inline (Este comando é específico de IPython/Jupyter e será removido ou comentado)

# Carregar o modelo treinado
model_path = 'c:\\Users\\pedro\\Downloads\\Senti-Pred\\src\\models\\sentiment_model.pkl' # Usando o caminho local
model = joblib.load(model_path)
print(f"Modelo carregado: {type(model).__name__}")

# Carregar os dados de teste
processed_path = 'c:\\Users\\pedro\\Downloads\\Senti-Pred\\data\\processed\\processed_data.csv' # Usando o caminho local
df = pd.read_csv(processed_path)

# Dividir em conjuntos de treino e teste (usando a mesma semente para consistência)

if 'text_lemmatized' in df.columns and 'sentiment' in df.columns:
    X = df['text_lemmatized']
    y = df['sentiment']
    
    # Usar a mesma divisão que foi usada no treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

## Avaliação Detalhada do Modelo

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular probabilidades (se o modelo suportar)
try:
    y_prob = model.predict_proba(X_test)
    has_probabilities = True
except:
    has_probabilities = False
    print("O modelo não suporta previsão de probabilidades.")

# Relatório de classificação detalhado
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.4f}")

# Matriz de confusão
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# Curvas ROC e Precision-Recall (para problemas binários)
if has_probabilities and len(model.classes_) == 2:
    # Curva ROC
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label='positive')
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    
    # Curva Precision-Recall
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1], pos_label='positive')
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.show()

## Análise de Erros

# Identificar exemplos classificados incorretamente
incorrect_indices = np.where(y_pred != y_test)[0]
incorrect_df = pd.DataFrame({
    'Texto': X_test.iloc[incorrect_indices],
    'Sentimento Real': y_test.iloc[incorrect_indices],
    'Sentimento Predito': y_pred[incorrect_indices]
})

# Exibir alguns exemplos de classificações incorretas
print(f"Total de classificações incorretas: {len(incorrect_indices)}")
if len(incorrect_indices) > 0:
    print(incorrect_df.head(10))

## Avaliação em Diferentes Segmentos de Dados

# Avaliar o desempenho por comprimento do texto
if 'text_lemmatized' in df.columns:
    # Calcular o comprimento do texto
    text_lengths = X_test.apply(lambda x: len(x.split()))
    
    # Definir categorias de comprimento
    bins = [0, 5, 10, 20, float('inf')]
    labels = ['Muito Curto', 'Curto', 'Médio', 'Longo']
    length_categories = pd.cut(text_lengths, bins=bins, labels=labels)
    
    # Calcular acurácia por categoria de comprimento
    accuracy_by_length = {}
    for category in labels:
        indices = length_categories == category
        if sum(indices) > 0:  # Verificar se há exemplos nesta categoria
            acc = accuracy_score(y_test[indices], y_pred[indices])
            accuracy_by_length[category] = acc
    
    # Visualizar
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_by_length.keys(), accuracy_by_length.values(), color='purple')
    plt.title('Acurácia por Comprimento de Texto')
    plt.xlabel('Categoria de Comprimento')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracy_by_length.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.show()

## Conclusões da Avaliação
# - Resumo do desempenho do modelo
# - Análise dos pontos fortes e fracos
# - Recomendações para melhorias futuras