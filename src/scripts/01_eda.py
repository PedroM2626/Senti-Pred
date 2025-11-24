# Análise Exploratória de Dados (EDA) - Senti-Pred
# Este notebook contém a análise exploratória dos dados para o projeto Senti-Pred, focado em análise de sentimentos.

# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
# %matplotlib inline (Este comando é específico de IPython/Jupyter e será removido ou comentado)

# Carregar os dados (procura automática em `data/raw` para ser portátil)
project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / 'data' / 'raw'

def find_raw_csv():
    # Preferências comuns, depois pega o primeiro .csv disponível
    candidates = ['test.csv', 'Test.csv', 'Test.CSV']
    for name in candidates:
        p = raw_dir / name
        if p.exists():
            return p
    files = list(raw_dir.glob('*.csv'))
    if files:
        return files[0]
    raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {raw_dir}")

data_path = find_raw_csv()
df = pd.read_csv(data_path)

# Exibir as primeiras linhas
df.head()

# Informações básicas sobre o dataset
print("Formato do dataset:", df.shape)
print("\nInformações do dataset:")
df.info()
print("\nEstatísticas descritivas:")
df.describe(include='all')

# Verificar valores ausentes
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Valores ausentes': missing_values,
    'Porcentagem (%)': missing_percentage
})

missing_df[missing_df['Valores ausentes'] > 0]

# Análise de distribuição de sentimentos (supondo que exista uma coluna 'sentiment')
if 'sentiment' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment', data=df)
    plt.title('Distribuição de Sentimentos')
    plt.xlabel('Sentimento')
    plt.ylabel('Contagem')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Análise de comprimento de texto (supondo que exista uma coluna 'text')
if 'Product_Description' in df.columns:
    df['text_length'] = df['Product_Description'].apply(len)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title('Distribuição do Comprimento de Texto')
    plt.xlabel('Comprimento do Texto')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()
    
    # Conclusões da Análise Exploratória
# - Resumo das principais descobertas
# - Insights para o pré-processamento
# - Direcionamentos para a modelagem