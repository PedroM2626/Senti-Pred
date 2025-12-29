#!/usr/bin/env python3
"""
Script para retreinar o modelo com Python 3.13 compatível
Executa o pipeline completo de treinamento e salva novo modelo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import nltk
import os
import sys

def setup_nltk():
    """Baixa pacotes necessários do NLTK"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def load_data():
    """Carrega dados de treinamento"""
    train_path = os.path.join('..', 'data', 'processed', 'processed_train.csv')
    
    if not os.path.exists(train_path):
        print(f"Erro: Arquivo {train_path} não encontrado!")
        print("Certifique-se de executar os scripts de processamento primeiro.")
        sys.exit(1)
    
    df = pd.read_csv(train_path)
    print(f"Dados carregados: {len(df)} amostras")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    return df

def create_pipeline():
    """Cria pipeline de processamento e modelo com melhores hiperparâmetros"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,  # Aumentado para capturar mais features
            ngram_range=(1, 3),  # Inclui trigramas
            stop_words='english',
            lowercase=True,
            max_df=0.85,  # Aumentado para reduzir overfitting
            min_df=3,     # Aumentado para remover ruído
            sublinear_tf=True  # Aplicar scaling sublinear
        )),
        ('classifier', LinearSVC(
            C=0.5,         # Reduzido para melhor generalização
            max_iter=2000,  # Aumentado para convergência
            random_state=42,
            class_weight='balanced'  # Balanceamento automático de classes
        ))
    ])
    return pipeline

def train_model(df):
    """Treina o modelo com melhorias no pré-processamento"""
    print("Treinando modelo com configurações otimizadas...")
    
    # **PRIORIZA text_lemmatized (mais processado) **
    if 'text_lemmatized' in df.columns:
        X = df['text_lemmatized']
        print("Usando coluna 'text_lemmatized' (mais processada)")
    elif 'text_clean' in df.columns:
        X = df['text_clean']
        print("Usando coluna 'text_clean'")
    elif 'text_no_stop' in df.columns:
        X = df['text_no_stop']
        print("Usando coluna 'text_no_stop'")
    else:
        print("Erro: Nenhuma coluna de texto processado encontrada!")
        print(f"Colunas disponíveis: {df.columns.tolist()}")
        sys.exit(1)
    
    y = df['sentiment']
    
    # **TRATAMENTO APERFEIÇOADO DE DADOS FALTANTES**
    print(f"Total inicial de amostras: {len(X)}")
    
    # Remove valores NaN e converte para string
    X = X.fillna('').astype(str)
    
    # Remove amostras com texto vazio ou muito curto
    mask = (X.str.strip() != '') & (X.str.len() > 2)
    X = X[mask]
    y = y[mask]
    
    print(f"Amostras após remover textos vazios: {len(X)}")
    
    # **ANÁLISE DE BALANCEAMENTO ANTES DO TREINAMENTO**
    print("\nDistribuição original das classes:")
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # **AUMENTA O CONJUNTO DE TREINO** (usa 90% para treino, 10% para validação)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y  # 90% treino, 10% validação
    )
    
    print(f"\nTamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de validação: {len(X_val)}")
    
    # **CRIA E TREINA PIPELINE OTIMIZADO**
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # **AVALIAÇÃO DETALHADA**
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n=== RESULTADOS ===")
    print(f"Acurácia no conjunto de validação: {accuracy:.4f}")
    print(f"Acurácia percentual: {accuracy*100:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(y_val, y_pred))
    
    return pipeline, accuracy, y_val, y_pred

def analyze_model_performance(pipeline, X_val, y_val, y_pred):
    """Análise detalhada do desempenho do modelo"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"\n=== ANÁLISE DETALHADA DE DESEMPENHO ===")
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precisão Macro: {precision:.4f}")
    print(f"Recall Macro: {recall:.4f}")
    print(f"F1-Score Macro: {f1:.4f}")
    
    # Análise por classe
    report = classification_report(y_val, y_pred, output_dict=True)
    print(f"\nDesempenho por classe:")
    for class_name in pipeline.classes_:
        if class_name in report:
            class_metrics = report[class_name]
            print(f"  {class_name}:")
            print(f"    Precisão: {class_metrics['precision']:.3f}")
            print(f"    Recall: {class_metrics['recall']:.3f}")
            print(f"    F1-Score: {class_metrics['f1-score']:.3f}")
            print(f"    Suporte: {int(class_metrics['support'])}")
    
    return accuracy, precision, recall, f1, report

def save_model(pipeline, accuracy, y_val, y_pred):
    """Salva modelo e métricas com análise completa"""
    # Cria diretórios
    os.makedirs('../src/models', exist_ok=True)
    os.makedirs('../reports/metrics', exist_ok=True)
    
    # Salva modelo
    model_path = '../src/models/sentiment_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"\nModelo salvo em: {model_path}")
    
    # Análise detalhada de performance
    accuracy_detailed, precision, recall, f1, report = analyze_model_performance(pipeline, None, y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    
    # Estrutura de métricas compatível com informações adicionais
    metrics = {
        'best_model': 'LinearSVC',
        'results': {
            'LinearSVC': {
                'accuracy': accuracy,
                'f1_macro': report['macro avg']['f1-score'],
                'precision_macro': precision,
                'recall_macro': recall,
                'train_time_seconds': 0,
                'predict_time_seconds': 0,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'model_parameters': {
                    'C': pipeline.named_steps['classifier'].C,
                    'max_iter': pipeline.named_steps['classifier'].max_iter,
                    'class_weight': 'balanced',
                    'features': pipeline.named_steps['tfidf'].max_features,
                    'ngram_range': pipeline.named_steps['tfidf'].ngram_range
                }
            }
        },
        'model_classes': pipeline.classes_.tolist(),
        'training_info': {
            'total_samples': len(y_val) + len(pipeline.named_steps['tfidf'].vocabulary_) if hasattr(pipeline.named_steps['tfidf'], 'vocabulary_') else len(y_val),
            'feature_count': len(pipeline.named_steps['tfidf'].vocabulary_) if hasattr(pipeline.named_steps['tfidf'], 'vocabulary_') else 0
        }
    }
    
    # Salva métricas
    metrics_path = '../reports/metrics/model_metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas salvas em: {metrics_path}")
    
    return model_path, metrics_path

def main():
    """Função principal"""
    print("=== Retreinamento do Modelo para Python 3.13 ===")
    print(f"Python: {sys.version}")
    
    # Setup
    setup_nltk()
    
    # Carrega dados
    df = load_data()
    
    # Treina modelo
    pipeline, accuracy, y_val, y_pred = train_model(df)
    
    # Salva modelo e métricas
    model_path, metrics_path = save_model(pipeline, accuracy, y_val, y_pred)
    
    print("\n=== Retreinamento Concluído ===")
    print(f"Modelo salvo: {model_path}")
    print(f"Métricas salvas: {metrics_path}")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nO modelo agora é compatível com Python 3.13!")

if __name__ == "__main__":
    main()