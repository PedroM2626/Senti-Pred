# Senti-Pred: Notebook Jupyter

Este guia detalha como executar o projeto Senti-Pred utilizando o notebook Jupyter `full_pipeline.ipynb`.

## Visão Geral

O notebook Jupyter `full_pipeline.ipynb` oferece uma abordagem interativa e passo a passo para explorar todo o pipeline de análise de sentimentos, desde a análise exploratória de dados (EDA) até o treinamento, avaliação e preparação para deploy do modelo. É ideal para experimentação, prototipagem e compreensão aprofundada de cada etapa.

## Estrutura do Notebook

O `full_pipeline.ipynb` está organizado nas seguintes seções:

-   **Configuração Inicial**: Importação de bibliotecas e configuração de variáveis de ambiente.
-   **Análise Exploratória de Dados (EDA)**: Visualização e entendimento dos dados.
-   **Pré-processamento de Texto**: Limpeza, tokenização e vetorização dos dados.
-   **Modelagem**: Treinamento de diferentes modelos de machine learning para classificação de sentimentos.
-   **Avaliação de Modelos**: Comparação e análise de desempenho dos modelos.
-   **Preparação para Deploy**: Exportação do modelo treinado para uso em produção.

## Instalação e Configuração

Certifique-se de ter seguido as etapas de instalação geral no [README.md](../README.md), incluindo a instalação das dependências e a configuração do ambiente virtual.

## Uso

1.  **Inicie o Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Navegue até o Notebook:**

    No seu navegador, navegue até o diretório `notebooks/` e abra o arquivo `full_pipeline.ipynb`.

3.  **Execute as Células:**

    Execute as células do notebook sequencialmente. Cada célula contém explicações detalhadas sobre a etapa que está sendo realizada.

    -   Você pode executar uma célula clicando nela e pressionando `Shift + Enter`.
    -   Para executar todas as células, vá em `Cell > Run All`.

## Saídas Geradas

Ao executar o notebook, os seguintes artefatos serão gerados:

-   **Dados Processados**: Arquivos CSV no diretório `data/processed/`.
-   **Visualizações**: Gráficos e imagens no diretório `reports/visualizacoes/`.
-   **Modelo Treinado**: O modelo de sentimento serializado (e.g., `.pkl`) no diretório `src/models/`.
-   **Relatório Técnico**: Um relatório detalhado em `reports/relatorio_tecnico.md`.

## Contribuição

Consulte o [README.md](../README.md) principal para diretrizes de contribuição.

## Licença

Consulte o [README.md](../README.md) principal para informações sobre a licença.