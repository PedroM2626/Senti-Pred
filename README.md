# Senti-Pred

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PedroM2626/Senti-Pred/blob/main/full_pipeline.ipynb)

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

## Materiais Complementares

1.  **Dashboard Interativo R Shiny**: Para análise exploratória de dados (EDA) interativa dos dados brutos, processados e de previsão.
    -   Veja o guia completo em: [README_r.md](README_r.md)
    -   **Deploy Online**: O dashboard está disponível publicamente em: [Senti-Pred EDA Dashboard](https://pedrom2626.shinyapps.io/r_shiny/)

2.  **Dashboard de Métricas Streamlit**: Para visualizar as métricas de avaliação do modelo de forma interativa.
    -   **Deploy Online**: O dashboard está disponível publicamente em: [Senti-Pred Metrics Dashboard](https://senti-pred-dashboard.streamlit.app/)
    -   **Execução Local (resumo)**:
        1.  Instale dependências: `pip install -r requirements.txt` (ou somente `pip install streamlit pandas pillow`).
        2.  Gere artefatos do pipeline:
            - `python src/scripts/02_preprocessing.py`
            - `python src/scripts/03_modeling.py`
            - Opcional: `python src/scripts/04_evaluation.py`
        3.  Navegue até `streamlit_dashboard` e execute: `streamlit run app.py`
        4.  Alternativamente, pela raiz: `streamlit run streamlit_dashboard/app.py`
    -   **Documentação Completa**: veja `streamlit_dashboard/README.md` para detalhes, estrutura esperada e troubleshooting.

3.  **Agente de Geração de Testes Unitários**: Para gerar automaticamente testes unitários para código Python usando IA.
    -   **Funcionalidade**: Gera testes unitários abrangentes usando Azure OpenAI e pytest
    -   **Documentação Completa**: veja `test_agent/README.md` para instruções detalhadas de uso e configuração.

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
├── streamlit_dashboard/
│   ├── app.py (Dashboard Streamlit)
│   └── README.md (Documentação do dashboard)
├── test_agent/
│   ├── agent.py (Agente de geração de testes)
│   ├── example_code.py (Código de exemplo)
│   ├── requirements.txt (Dependências do agente)
│   └── README.md (Documentação do agente)
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
