"""
Módulo para treinamento e predição de modelos de análise de sentimentos.
"""
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


class SentimentModel:
    """Classe para gerenciar o modelo de análise de sentimentos."""
    
    def __init__(self, model_path=None):
        """
        Inicializa o modelo de sentimentos.
        
        Args:
            model_path (str, optional): Caminho para o modelo salvo. Se None, cria um novo modelo.
        """
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', LogisticRegression(random_state=42, max_iter=1000))
            ])
    
    def train(self, X_train, y_train):
        """
        Treina o modelo com os dados fornecidos.
        
        Args:
            X_train: Textos de treinamento
            y_train: Rótulos de sentimento
            
        Returns:
            self: O modelo treinado
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, texts):
        """
        Faz predições de sentimento para os textos fornecidos.
        
        Args:
            texts: Lista ou série de textos para predição
            
        Returns:
            np.array: Array com as predições de sentimento
        """
        return self.model.predict(texts)
    
    def predict_proba(self, texts):
        """
        Retorna as probabilidades de cada classe para os textos fornecidos.
        
        Args:
            texts: Lista ou série de textos para predição
            
        Returns:
            np.array: Array com as probabilidades de cada classe
        """
        return self.model.predict_proba(texts)
    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo nos dados de teste.
        
        Args:
            X_test: Textos de teste
            y_test: Rótulos de sentimento reais
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def save(self, model_path):
        """
        Salva o modelo treinado no caminho especificado.
        
        Args:
            model_path (str): Caminho para salvar o modelo
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Modelo salvo em: {model_path}")


def train_model(data_path, model_save_path):
    """
    Função para treinar e salvar um modelo a partir de um arquivo de dados.
    
    Args:
        data_path (str): Caminho para o arquivo de dados processados
        model_save_path (str): Caminho para salvar o modelo treinado
        
    Returns:
        SentimentModel: O modelo treinado
    """
    # Carregar dados
    df = pd.read_csv(data_path)
    
    # Verificar se as colunas necessárias existem
    if 'text_lemmatized' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("O arquivo de dados deve conter as colunas 'text_lemmatized' e 'sentiment'")
    
    # Dividir em treino e teste
    from sklearn.model_selection import train_test_split
    X = df['text_lemmatized']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = SentimentModel()
    model.train(X_train, y_train)
    
    # Avaliar modelo
    eval_results = model.evaluate(X_test, y_test)
    print(f"Acurácia do modelo: {eval_results['accuracy']:.4f}")
    
    # Salvar modelo
    model.save(model_save_path)
    
    return model


if __name__ == "__main__":
    # Exemplo de uso
    # Download latest version
    data_path = 'c:\\Users\\pedro\\Downloads\\Senti-Pred\\data\\raw\\Test.csv'
    model_path = "./sentiment_model.pkl"
    
    try:
        model = train_model(data_path, model_path)
        print("Modelo treinado e salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao treinar o modelo: {str(e)}")