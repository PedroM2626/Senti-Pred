# Dashboard de Métricas — Streamlit

Este README documenta o dashboard em Streamlit que exibe as métricas e visualizações geradas pelo pipeline Senti-Pred.

**Deploy Online**: O dashboard está disponível publicamente em: [Senti-Pred Metrics Dashboard](https://senti-pred-dashboard.streamlit.app/)

## Visão Geral

- O dashboard lê o arquivo `reports/metrics/model_metrics.json` e mostra um resumo do melhor modelo e uma tabela com as métricas por algoritmo.
- Exibe imagens geradas pelo pipeline em `reports/visualizacoes/` (curvas ROC, PR e matrizes de confusão comparativas).
- Arquivo principal: `streamlit_dashboard/app.py`.

## Pré-requisitos

- Python 3.8+
- Pip
- Dependências do projeto instaladas

Instale as dependências usando o `requirements.txt` na raiz do projeto:

```bash
pip install -r requirements.txt
```

Alternativamente, apenas para o dashboard:

```bash
pip install streamlit pandas pillow
```

## Preparação dos Artefatos (métricas e imagens)

O dashboard depende de arquivos gerados pelos scripts do pipeline. Execute em ordem:

1. Pré-processamento
   ```bash
   python src/scripts/02_preprocessing.py
   ```
   - Gera `data/processed/processed_data.pkl`.

2. Modelagem e geração de métricas/visuais
   ```bash
   python src/scripts/03_modeling.py
   ```
   - Gera `reports/metrics/model_metrics.json` e imagens em `reports/visualizacoes/`.

3. Avaliação (opcional, complementa métricas e imagens)
   ```bash
   python src/scripts/04_evaluation.py
   ```
   - Atualiza `reports/metrics/model_metrics.json` e cria imagens adicionais (ex.: `evaluation_confusion_matrix.png`, `evaluation_roc_pr.png`).

## Como Executar o Dashboard

Execute a partir do diretório `streamlit_dashboard`:

```bash
cd streamlit_dashboard
streamlit run app.py
```

Se preferir executar a partir da raiz do projeto:

```bash
streamlit run streamlit_dashboard/app.py
```

Abra o endereço local exibido no terminal (ex.: `http://localhost:8501`).

## Estrutura Esperada

```
Senti-Pred/
├── streamlit_dashboard/
│   └── app.py
├── reports/
│   ├── metrics/
│   │   └── model_metrics.json
│   └── visualizacoes/
│       ├── comparison_roc.png
│       ├── comparison_pr.png
│       └── comparison_confusion_matrices.png
└── data/
    └── processed/
        └── processed_data.pkl
```

## Mensagens e Erros Comuns

- "Arquivo de métricas não encontrado": execute `03_modeling.py` e/ou `04_evaluation.py` para gerar `reports/metrics/model_metrics.json`.
- "Visualização não encontrada": verifique se os PNGs existem em `reports/visualizacoes/`. Rode `03_modeling.py` para gerar comparativos.
- Permissões/paths: assegure-se de executar os comandos a partir da raiz do projeto ou ajuste os caminhos conforme indicado acima.

## Customização

- Edite `streamlit_dashboard/app.py` para alterar colunas exibidas, legendas e caminhos das imagens.
- Os gráficos são salvos pelos scripts em `reports/visualizacoes/`. Para incluir novos gráficos, gere-os nos scripts e referencie-os no dashboard.

## Deploy com Docker (Recomendado)

O projeto já está configurado para ser executado via Docker, o que garante que todas as dependências e caminhos funcionem corretamente sem precisar configurar o ambiente local manualmente.

1. Certifique-se de ter o Docker e Docker Compose instalados.

2. Na raiz do projeto (`Senti-Pred/`), execute:

```bash
docker-compose up --build dashboard
```

3. Acesse o dashboard em `http://localhost:8501`.

## Deploy no Streamlit Cloud

Para deployar publicamente no Streamlit Cloud:

1. Suba este repositório para o GitHub.
2. Crie uma conta no [Streamlit Cloud](https://share.streamlit.io/).
3. Conecte sua conta do GitHub e selecione o repositório.
4. Configure:
   - **Main file path:** `streamlit_dashboard/app.py`
   - **Python version:** 3.9 (ou compatível)
5. Clique em **Deploy**.

**Nota:** Como o dashboard depende de arquivos gerados localmente (`reports/`), você deve garantir que esses arquivos (`model_metrics.json`, imagens png) estejam commitados no repositório ou sejam gerados como parte do processo de build (o que é mais complexo no Streamlit Cloud gratuito). A opção mais simples para este projeto específico é commitar a pasta `reports/` gerada.

## Licença

Este dashboard segue a mesma licença do repositório principal (MIT).