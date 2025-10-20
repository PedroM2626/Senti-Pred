# Senti-Pred: GCP Vertex AI + Docker

Este guia detalha como implantar o projeto Senti-Pred no Google Cloud Platform (GCP) utilizando o Vertex AI e Docker.

## Visão Geral

Esta abordagem é voltada para a implantação em ambientes de produção, aproveitando a escalabilidade e os recursos gerenciados do GCP. O modelo de análise de sentimentos será empacotado em um contêiner Docker e implantado como um endpoint na Vertex AI, permitindo o consumo via API.

## Pré-requisitos

Além dos pré-requisitos gerais listados no [README.md](../README.md), você precisará:

-   **Conta GCP**: Uma conta ativa no Google Cloud Platform.
-   **Projeto GCP**: Um projeto GCP configurado com faturamento ativado.
-   **Google Cloud SDK (gcloud)**: Instalado e configurado para o seu projeto GCP.
-   **Docker**: Instalado e em execução na sua máquina local.
-   **Permissões**: As permissões necessárias no GCP para criar e gerenciar recursos da Vertex AI, Cloud Storage e Container Registry.

## Instalação e Configuração

Certifique-se de ter seguido as etapas de instalação geral no [README.md](../README.md).

### 1. Autenticar no GCP

Certifique-se de que o `gcloud` esteja autenticado e configurado para o seu projeto:

```bash
gcloud auth login
gcloud config set project SEU_PROJETO_GCP
```

### 2. Baixar o Modelo Treinado do GCP

O modelo treinado é armazenado em um bucket do Google Cloud Storage. Utilize o comando `gsutil cp` para baixá-lo para o seu ambiente local:

```bash
gsutil cp gs://seu-bucket-gcp/caminho/para/sentiment_model.pkl src/models/sentiment_model.pkl
```

**Nota**: Substitua `seu-bucket-gcp/caminho/para/sentiment_model.pkl` pelo caminho real do seu modelo no Cloud Storage.

### 3. Construir a Imagem Docker

Navegue até o diretório raiz do projeto e construa a imagem Docker. O `Dockerfile` está configurado para empacotar a aplicação Django e o modelo de sentimento.

```bash
docker build -t gcr.io/SEU_PROJETO_GCP/senti-pred-api:latest .
```

**Nota**: Substitua `SEU_PROJETO_GCP` pelo ID do seu projeto GCP.

### 4. Enviar a Imagem Docker para o Google Container Registry

```bash
docker push gcr.io/SEU_PROJETO_GCP/senti-pred-api:latest
```

### 5. Criar um Endpoint na Vertex AI

Crie um endpoint na Vertex AI para servir o modelo. Isso pode ser feito via console GCP ou via `gcloud`.

```bash
gcloud ai endpoints create --display-name="senti-pred-endpoint" --region=us-central1
```

Anote o `ENDPOINT_ID` retornado.

### 6. Implantar o Modelo no Endpoint da Vertex AI

Implante o modelo no endpoint criado, referenciando a imagem Docker que você enviou para o Container Registry.

```bash
gcloud ai endpoints deploy-model ENDPOINT_ID \
    --display-name="senti-pred-model" \
    --model-id="senti-pred-model-id" \
    --container-image-uri="gcr.io/SEU_PROJETO_GCP/senti-pred-api:latest" \
    --region=us-central1 \
    --machine-type=n1-standard-2 \
    --min-replica-count=1 \
    --max-replica-count=1
```

**Nota**: Substitua `ENDPOINT_ID` pelo ID do endpoint e `SEU_PROJETO_GCP` pelo ID do seu projeto GCP. Ajuste o `machine-type` conforme necessário.

## Uso da API Implantada

Após a implantação, você poderá enviar requisições para o endpoint da Vertex AI para obter predições de sentimento. A URL do endpoint estará disponível no console da Vertex AI.

Exemplo de requisição (usando `curl`):

```bash
curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    https://SEU_REGION-aiplatform.googleapis.com/v1/projects/SEU_PROJETO_GCP/locations/SEU_REGION/endpoints/ENDPOINT_ID:predict \
    -d "{\"instances\": [{\"text\": \"Este é um ótimo produto!\"}]}"
```

**Nota**: Substitua `SEU_REGION`, `SEU_PROJETO_GCP` e `ENDPOINT_ID` pelos valores corretos.

## Contribuição

Consulte o [README.md](../README.md) principal para diretrizes de contribuição.

## Licença

Consulte o [README.md](../README.md) principal para informações sobre a licença.