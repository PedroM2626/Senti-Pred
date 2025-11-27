# Senti-Pred: Scripts Python Modulares

Este guia detalha como executar o projeto Senti-Pred utilizando os scripts Python modulares localizados no diretório `src/`.

## Visão Geral

Esta abordagem é ideal para quem busca uma execução mais controlada, modular e que pode ser facilmente integrada em outros sistemas ou fluxos de trabalho automatizados. Os scripts cobrem desde o pré-processamento de dados até o treinamento do modelo e a disponibilização de uma API para predições.

## Estrutura dos Scripts Principais

-   `src/scripts/01_data_ingestion.py`: Responsável pela ingestão inicial dos dados.
-   `src/scripts/02_preprocessing.py`: Realiza o pré-processamento e limpeza dos dados.
-   `src/models/sentiment_model.py`: Contém a lógica para treinamento e predição do modelo de sentimento.
-   `src/scripts/04_evaluation.py`: Avalia o desempenho do modelo treinado.
-   `src/api/views.py`: Define os endpoints da API Django para predição de sentimentos.
-   `src/api/urls.py`: Configura as rotas da API.
-   `src/api/settings.py`: Configurações do projeto Django.

## Instalação e Configuração

Certifique-se de ter seguido as etapas de instalação geral no [README.md](../README.md), incluindo a instalação das dependências e a configuração do ambiente virtual.

## Uso

### 1. Executar o Pipeline Completo (Passo a Passo)

Você pode executar cada etapa do pipeline sequencialmente:

#### Ingestão de Dados

```bash
python src/scripts/01_data_ingestion.py
```

#### Pré-processamento de Dados

```bash
python src/scripts/02_preprocessing.py
```

#### Treinamento do Modelo

```bash
python src/models/sentiment_model.py
```

#### Avaliação do Modelo

```bash
python src/scripts/04_evaluation.py
```

### 2. Iniciar a API Django

Para disponibilizar o modelo treinado via API:

```bash
cd src/api
python manage.py runserver
```

A API estará disponível em `http://localhost:8000/api/predict/`.

Você pode testar a API enviando uma requisição POST, por exemplo, usando `curl` ou Postman:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "Este é um ótimo produto!"}' http://localhost:8000/api/predict/
```

### 3. Executar Testes

Para garantir a qualidade do código, execute os testes:

#### Testes Unitários

```bash
pytest tests/unit/
```

#### Testes de Integração

```bash
pytest tests/integration/
```

#### Testes de Aceitação

```bash
pytest tests/acceptance/
```

## Contribuição

Consulte o [README.md](../README.md) principal para diretrizes de contribuição.

## Licença

Consulte o [README.md](../README.md) principal para informações sobre a licença.