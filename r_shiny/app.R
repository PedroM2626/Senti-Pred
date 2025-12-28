library(shiny)
library(ggplot2)
library(dplyr)
library(readr)
library(DT)
library(plotly)
library(shinydashboard)
library(jsonlite)

# UI
# Definição da interface (UI) do dashboard: layout, menu e abas principais
ui <- dashboardPage(
  dashboardHeader(title = "Senti-Pred EDA Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dados Brutos", tabName = "raw_data", icon = icon("database")),
      menuItem("Visualizações", tabName = "visualizations", icon = icon("chart-area")),
      menuItem("Métricas / Previsões", tabName = "metrics", icon = icon("chart-line"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .box {
          margin-bottom: 20px !important;
        }
        .content-wrapper {
          padding-bottom: 30px;
        }
        .box .box-body {
          overflow: hidden;
        }
        .shiny-image-output {
          text-align: center;
        }
        .shiny-image-output img {
          max-width: 100%;
          width: 100%;
          height: auto;
          display: block;
        }
      "))
    ),
    tabItems(
    # Raw Data Tab
    # Exibe status e amostra dos dados brutos (treino/validação) lidos de data/raw/
    tabItem(tabName = "raw_data",
              h2("Análise de Dados Brutos"),
              textOutput("raw_data_status"),
              fluidRow(
                box(title = "Amostra dos Dados Brutos (treino + validação)", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("raw_data_table")
                )
              )
      ),

      # Visualizations Tab
      # Mostra imagens geradas pelo pipeline (EDA e métricas comparativas)
      tabItem(tabName = "visualizations",
              h2("Visualizações Geradas pelo Pipeline"),
              fluidRow(
                box(title = "Distribuição de Comprimento de Texto", status = "info", solidHeader = TRUE, width = 6,
                    imageOutput('img_text_length', width = "100%", height = "380px"),
                    br(), br()
                ),
                box(title = "Top Words (raw)", status = "info", solidHeader = TRUE, width = 6,
                    imageOutput('img_top_words', width = "100%", height = "380px"),
                    br(), br()
                )
              ),
              fluidRow(
                box(title = "ROC Comparativo", status = "primary", solidHeader = TRUE, width = 6,
                    imageOutput('img_roc', width = "100%", height = "380px"),
                    br(), br()
                ),
                box(title = "PR Comparativo", status = "primary", solidHeader = TRUE, width = 6,
                    imageOutput('img_pr', width = "100%", height = "380px"),
                    br(), br()
                )
              ),
              fluidRow(
                box(title = "Matrizes de Confusão Comparativas", status = "primary", solidHeader = TRUE, width = 12,
                    imageOutput('img_cm', width = "100%", height = "480px"),
                    br(), br()
                )
              )
      ),

      # Metrics / Predictions Tab
      # Apresenta o JSON de métricas e tabela de previsões (se existir)
      tabItem(tabName = "metrics",
              h2("Métricas e Previsões"),
              textOutput("metrics_status"),
              fluidRow(
                box(title = "Métricas (JSON)", status = "primary", solidHeader = TRUE, width = 12,
                    verbatimTextOutput('metrics_json')
                )
              ),
              fluidRow(
                box(title = "Tabela de Previsões (se existir)", status = "info", solidHeader = TRUE, width = 12,
                    DTOutput('predictions_table')
                )
              )
      )
    )
  )
)

# Server
# Lógica do backend: carrega dados, renderiza tabelas e imagens,
# lê métricas/JSON e exibe previsões quando disponíveis
server <- function(input, output) {
  # Paths
  # Caminhos para arquivos e diretórios utilizados durante a execução
  train_path <- file.path('data', 'raw', 'twitter_training.csv')
  val_path <- file.path('data', 'raw', 'twitter_validation.csv')
  vis_dir <- file.path('reports', 'visualizacoes')
  metrics_path <- file.path('reports', 'metrics', 'model_metrics.json')
  predictions_path <- file.path('data', 'processed', 'predictions.csv')

  # Carregamento dos dados brutos (treino e validação)
  # Lê CSVs em data/raw/, combina quando ambos existem e marca a coluna 'split'
  raw_data <- reactive({
    df_train <- tryCatch(read_csv(train_path, col_names = c('tweet_id','entity','sentiment','text'), show_col_types = FALSE), error = function(e) NULL)
    df_val <- tryCatch(read_csv(val_path, col_names = c('tweet_id','entity','sentiment','text'), show_col_types = FALSE), error = function(e) NULL)
    if (!is.null(df_train) && !is.null(df_val)) {
      df <- bind_rows(df_train %>% mutate(split='train'), df_val %>% mutate(split='validation'))
    } else if (!is.null(df_train)) {
      df <- df_train %>% mutate(split='train')
    } else if (!is.null(df_val)) {
      df <- df_val %>% mutate(split='validation')
    } else {
      df <- tibble()
    }
    df
  })

  # Status textual sobre presença dos dados brutos e quantidade de linhas
  output$raw_data_status <- renderText({
    df <- raw_data()
    if (nrow(df) == 0) {
      "Dados brutos: Não encontrados em data/raw/. Coloque os CSVs 'twitter_training.csv' e 'twitter_validation.csv'."
    } else {
      paste0('Dados brutos carregados: ', nrow(df), ' linhas')
    }
  })

  # Exibe uma amostra (até 200 linhas) dos dados brutos carregados
  output$raw_data_table <- renderDT({
    df <- raw_data()
    if (nrow(df) == 0) return(datatable(data.frame(message='Nenhum dado disponível')))
    datatable(head(df, 200))
  })

  # Visualizações: imagens geradas pelo pipeline
  # Helper para servir imagens geradas pelo pipeline; retorna NULL se não existir
  image_or_placeholder <- function(path) {
    if (file.exists(path)) {
      list(src = path, contentType = 'image/png', alt = basename(path))
    } else {
      NULL
    }
  }

  # Distribuição de comprimento de texto (EDA)
  output$img_text_length <- renderImage({
    path <- file.path(vis_dir, 'text_length.png')
    image_or_placeholder(path)
  }, deleteFile = FALSE)

  # Palavras mais frequentes (dados brutos)
  output$img_top_words <- renderImage({
    path <- file.path(vis_dir, 'top_words_raw.png')
    image_or_placeholder(path)
  }, deleteFile = FALSE)

  # Curvas ROC comparativas entre modelos
  output$img_roc <- renderImage({
    path <- file.path(vis_dir, 'comparison_roc.png')
    image_or_placeholder(path)
  }, deleteFile = FALSE)

  # Curvas Precision-Recall comparativas entre modelos
  output$img_pr <- renderImage({
    path <- file.path(vis_dir, 'comparison_pr.png')
    image_or_placeholder(path)
  }, deleteFile = FALSE)

  # Matrizes de confusão comparativas entre modelos
  output$img_cm <- renderImage({
    path <- file.path(vis_dir, 'comparison_confusion_matrices.png')
    image_or_placeholder(path)
  }, deleteFile = FALSE)

  # Métricas do modelo (JSON) — carrega 'reports/metrics/model_metrics.json'
  metrics <- reactive({
    if (!file.exists(metrics_path)) return(NULL)
    tryCatch(fromJSON(metrics_path), error = function(e) NULL)
  })

  # Status textual das métricas (melhor modelo, ou aviso se faltarem)
  output$metrics_status <- renderText({
    if (is.null(metrics())) "Métricas: não encontradas (execute '03_modeling.py' e '04_evaluation.py')" else paste0('Métricas carregadas — melhor modelo: ', metrics()$best_model)
  })

  # Impressão completa do JSON de métricas para inspeção
  output$metrics_json <- renderPrint({
    m <- metrics()
    if (is.null(m)) return(cat('Nenhuma métrica disponível'))
    print(m)
  })

  # Tabela de previsões (se existir 'data/processed/predictions.csv')
  output$predictions_table <- renderDT({
    if (!file.exists(predictions_path)) return(datatable(data.frame(message='Nenhum arquivo de previsões encontrado (data/processed/predictions.csv)')))
    df <- tryCatch(read_csv(predictions_path, show_col_types = FALSE), error = function(e) data.frame())
    if (nrow(df) == 0) return(datatable(data.frame(message='Arquivo de previsões vazio ou inválido')))
    datatable(df)
  })
}

# Inicialização do aplicativo Shiny
shinyApp(ui = ui, server = server)
