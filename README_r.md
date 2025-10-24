# Senti-Pred: Dashboard Interativo com R Shiny

Este guia detalha como configurar e executar o dashboard interativo de Análise Exploratória de Dados (EDA) desenvolvido em R Shiny para o projeto Senti-Pred.

## Visão Geral

O dashboard R Shiny oferece uma interface interativa para explorar os dados brutos, processados e as previsões geradas pelo modelo de análise de sentimentos. Ele permite visualizar distribuições, tendências e outras métricas importantes de forma dinâmica, facilitando a compreensão e a validação dos dados e resultados.

## Estrutura do Projeto R

```
senti-pred/
├── r_shiny/
│   ├── app.R (ou ui.R e server.R)
│   └── data/
│       ├── raw/ (dados brutos para R)
│       ├── processed/ (dados processados para R)
│       └── predictions/ (dados de previsão para R)
└── ...
```

## Pré-requisitos

Além dos pré-requisitos gerais listados no [README.md](../README.md), você precisará:

-   **R**: Instale o ambiente de programação R (https://www.r-project.org/).
-   **RStudio (Opcional, mas Recomendado)**: Uma IDE para desenvolvimento em R (https://www.rstudio.com/).
-   **Pacotes R**: Os seguintes pacotes R precisarão ser instalados:
    -   `shiny`
    -   `ggplot2`
    -   `dplyr`
    -   `readr`
    -   `DT`
    -   `plotly`
    -   `shinydashboard` (se for usar o layout de dashboard)

## Instalação dos Pacotes R

Abra o console do R (ou RStudio) e execute os seguintes comandos para instalar os pacotes necessários:

```R
install.packages(c("shiny", "ggplot2", "dplyr", "readr", "DT", "plotly", "shinydashboard"))
```

## Uso

### 1. Preparar os Dados

Certifique-se de que os dados brutos, processados e de previsão estejam disponíveis nos diretórios apropriados dentro de `r_shiny/data/`. Você pode precisar copiar ou gerar esses arquivos a partir do pipeline Python.

### 2. Executar o Dashboard Shiny

Navegue até o diretório `r_shiny/` no seu terminal e execute o aplicativo R Shiny:

```bash
R -e "shiny::runApp()"
```

Alternativamente, se estiver usando RStudio, abra o arquivo `app.R` (ou `ui.R`/`server.R`) e clique no botão "Run App" no canto superior direito.

O dashboard será aberto automaticamente no seu navegador padrão.

## Contribuição

Consulte o [README.md](../README.md) principal para diretrizes de contribuição.

## Licença

Consulte o [README.md](../README.md) principal para informações sobre a licença.