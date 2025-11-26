# Senti-Pred

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedroM2626/Senti-Pred/blob/main/notebooks/full_pipeline.ipynb)

Projeto de análise de sentimentos com processamento de linguagem natural e aprendizado de máquina.

## Visão Geral

O Senti-Pred é um projeto de análise de sentimentos que utiliza técnicas de processamento de linguagem natural e aprendizado de máquina para classificar textos de acordo com o sentimento expresso. Este repositório oferece diferentes abordagens para explorar e utilizar o pipeline de análise de sentimentos.

## Fonte dos Dados

O dataset utilizado neste projeto foi obtido a partir do repositório público no Kaggle:

https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

Por favor, verifique os termos de uso e a licença do dataset no Kaggle antes de redistribuir os dados.

## Opções de Execução

O projeto pode ser executado de três maneiras principais, cada uma com sua própria documentação detalhada:

1.  **Scripts Python Modulares**: Para uma execução mais controlada e modular, ideal para integração em outros sistemas ou automação.
    -   Os scripts atualizados estão em `src/scripts/` e seguem a divisão em etapas:
        - `01_eda.py` — Análise exploratória e geração de PNGs em `reports/visualizacoes/`.
        - `02_preprocessing.py` — Pré-processamento em inglês; gera um artefato binário em `data/processed/processed_data.pkl`.
        - `03_modeling.py` — Treina modelos a partir do pickle processado, salva o melhor em `src/models/sentiment_model.pkl`, gera métricas JSON e gráficos comparativos (ROC/PR/confusion) em `reports/visualizacoes/`.
        - `04_evaluation.py` — Avaliação do modelo salvo; gera imagens e `reports/metrics/model_metrics.json`.
    -   Veja o guia completo em: [README_scripts.md](README_scripts.md)

2.  **Notebook Jupyter**: Para uma exploração interativa e desenvolvimento passo a passo do pipeline completo.
    -   Veja o guia completo em: [README_jupyter.md](README_jupyter.md)

3.  **GCP Vertex AI + Docker**: Para deploy em ambiente de produção utilizando a infraestrutura do Google Cloud Platform e Docker.
    -   Veja o guia completo em: [README_gcp.md](README_gcp.md)

4.  **Dashboard Interativo R Shiny**: Para análise exploratória de dados (EDA) interativa dos dados brutos, processados e de previsão.
    -   Veja o guia completo em: [README_r.md](README_r.md)
    -   **Deploy Online**: O dashboard está disponível publicamente em: <mcurl name="https://pedrom2626.shinyapps.io/r_shiny/" url="https://pedrom2626.shinyapps.io/r_shiny/"></mcurl>

5.  **Dashboard de Métricas Streamlit**: Para visualizar as métricas de avaliação do modelo de forma interativa.
    -   **Execução Local**:
        1.  Certifique-se de ter o Streamlit instalado (`pip install streamlit`).
        2.  Navegue até o diretório `streamlit_dashboard`: `cd streamlit_dashboard`
        3.  Execute o dashboard: `streamlit run app.py`

## Estrutura do Projeto

```
senti-pred/
├── README.md
├── README_scripts.md
├── README_jupyter.md
├── README_gcp.md
├── README_r.md
├── data/
│   ├── raw/ (dados originais)
│   └── processed/ (dados processados)
├── r_shiny/
│   └── app.R (Aplicativo R Shiny)
├── full_pipeline.ipynb (Pipeline completo) 
├── src/
│   ├── models/ (código dos modelos)
│   ├── api/ (Django API)
│   └── scripts/ (Scripts de processamento)
├── reports/
│   ├── relatorio_tecnico.md
│   └── visualizacoes/ (gráficos e visualizações)
├── .env.example
├── .env
├── .gitignore
├── requirements.txt
└── docker-compose.yml
```

## Instalação e Configuração (Geral)

### Pré-requisitos

-   Python 3.8+
-   Pip (gerenciador de pacotes Python)
-   Docker e Docker Compose (para deploy com Docker)
-   Virtualenv (opcional, mas recomendado)

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/PedroM2626/Senti-Pred.git
    cd senti-pred
    ```

2.  Crie e ative um ambiente virtual (opcional):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure as variáveis de ambiente:
    ```bash
    cp .env.example .env
    # Edite o arquivo .env com suas configurações
    ```

## Contribuição

1.  Faça um fork do projeto
2.  Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3.  Faça commit das suas alterações (`git commit -am 'Adiciona nova feature'`)
4.  Faça push para a branch (`git push origin feature/nova-feature`)
5.  Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Agente de Geração de Testes Unitários

Este projeto inclui um agente de IA que gera testes unitários para código Python usando a biblioteca `pytest` e o Azure OpenAI.

### Como Usar o Agente de Testes

1.  **Configurar Variáveis de Ambiente**:
    Crie um arquivo `.env` no diretório `test_agent` (se ainda não existir) e adicione as seguintes variáveis, substituindo os valores pelos seus dados do Azure OpenAI:

    ```
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    AZURE_OPENAI_DEPLOYMENT_NAME=your_azure_openai_deployment_name
    ```

    Você pode usar o arquivo `.env.example` como base.

2.  **Preparar o Código para Testar**:
    Coloque o código Python para o qual você deseja gerar testes no arquivo `example_code.py` dentro do diretório `test_agent`.

3.  **Executar o Agente**:
    Navegue até o diretório `test_agent` e execute o script `agent.py`:

    ```bash
    cd test_agent
    python agent.py
    ```

    O agente irá gerar um arquivo de testes chamado `test_example_code.py` no mesmo diretório.

4.  **Rodar os Testes Gerados**:
    Para executar os testes gerados, certifique-se de ter o `pytest` instalado (`pip install pytest`) e execute:

    ```bash
    pytest test_example_code.py
    ```