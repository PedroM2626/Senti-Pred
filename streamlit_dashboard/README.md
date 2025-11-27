# Dashboard de Métricas — Streamlit

Este README documenta o dashboard em Streamlit que exibe as métricas e visualizações geradas pelo pipeline Senti-Pred.

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

## Licença

Este dashboard segue a mesma licença do repositório principal (MIT).