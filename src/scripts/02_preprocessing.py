# Pré-processamento de Dados - Senti-Pred
# Este notebook contém as etapas de pré-processamento dos dados para o projeto Senti-Pred.

# Importações necessárias
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Download dos recursos do NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('rslp', quiet=True)

# Carregar os dados
data_path = r'c:\Users\pedro\Downloads\Senti-Pred\data\raw\Test.csv' # Usando o caminho local
df = pd.read_csv(data_path)

# Exibir as primeiras linhas
df.head()

# Funções de pré-processamento
def clean_text(text):
    if isinstance(text, str):
        # Converter para minúsculas
        text = text.lower()
        # Remover URLs
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        # Remover menções e hashtags
        text = re.sub(r'@\\w+|#\\w+', '', text)
        # Remover pontuação e caracteres especiais
        text = re.sub(r'[^\\w\\s]', '', text)
        # Remover números
        text = re.sub(r'\\d+', '', text)
        # Remover espaços extras
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    return ''

def remove_stopwords(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('portuguese'))
        word_tokens = word_tokenize(text, language='portuguese')
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_text)
    return ''

def lemmatize_text(text):
    if isinstance(text, str):
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text, language='portuguese')
        lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    return ''

def assign_sentiment(text):
    if isinstance(text, str):
        text = text.lower()
        positive_keywords = ['bom', 'ótimo', 'excelente', 'incrível', 'feliz', 'gostei', 'amo', 'recomendo']
        negative_keywords = ['ruim', 'péssimo', 'terrível', 'odeio', 'não gostei', 'decepcionado', 'problema']
        
        if any(word in text for word in positive_keywords):
            return 'positive'
        elif any(word in text for word in negative_keywords):
            return 'negative'
    return 'neutral'

# Aplicar pré-processamento
if 'Product_Description' in df.columns:
    # Criar cópia do dataframe original
    df_processed = df.copy()
    
    # Aplicar limpeza de texto
    df_processed['text_clean'] = df_processed['Product_Description'].apply(clean_text)
    
    # Remover stopwords
    df_processed['text_no_stopwords'] = df_processed['text_clean'].apply(remove_stopwords)
    
    # Lematização
    df_processed['text_lemmatized'] = df_processed['text_no_stopwords'].apply(lemmatize_text)
    
    # Atribuir sentimento
    df_processed['sentiment'] = df_processed['Product_Description'].apply(assign_sentiment)
    
    # Salvar dados processados
    processed_path = r'c:\Users\pedro\Downloads\Senti-Pred\data\processed\processed_data.csv' # Usando o caminho local
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"Dados processados salvos em: {processed_path}")

    # Análise dos Dados Processados
    # Vamos analisar algumas estatísticas dos dados após o pré-processamento.

    if 'Product_Description' in df.columns and 'text_lemmatized' in df_processed.columns:
        # Comprimento dos textos antes e depois do processamento
        df_processed['original_length'] = df_processed['Product_Description'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        df_processed['processed_length'] = df_processed['text_lemmatized'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Visualizar a redução no comprimento dos textos
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df_processed['original_length'], bins=30, kde=True, color='blue')
        plt.title('Distribuição do Comprimento Original')
        plt.xlabel('Número de Palavras')
        plt.ylabel('Frequência')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df_processed['processed_length'], bins=30, kde=True, color='green')
        plt.title('Distribuição do Comprimento Após Processamento')
        plt.xlabel('Número de Palavras')
        
        plt.tight_layout()
        plt.show()
        
        # Estatísticas da redução
        reduction = ((df_processed['original_length'] - df_processed['processed_length']) / df_processed['original_length']) * 100
        print(f"Redução média no comprimento do texto: {reduction.mean():.2f}%")

    # Conclusões do Pré-processamento
    # - Resumo das transformações aplicadas
    # - Impacto do pré-processamento nos dados
    # - Próximos passos para a modelagem
else:
    print("Erro: A coluna 'Product_Description' não foi encontrada no DataFrame. Certifique-se de que o arquivo CSV contém a coluna 'Product_Description'.")
    sys.exit(1)