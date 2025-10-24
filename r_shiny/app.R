library(shiny)
library(ggplot2)
library(dplyr)
library(readr)
library(DT)
library(plotly)
library(shinydashboard)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Senti-Pred EDA Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dados Brutos", tabName = "raw_data", icon = icon("database")),
      menuItem("Dados Processados", tabName = "processed_data", icon = icon("cogs")),
      menuItem("Previsões", tabName = "predictions", icon = icon("chart-line"))
    )
  ),
  dashboardBody(
    tabItems(
      # Raw Data Tab
      tabItem(tabName = "raw_data",
              h2("Análise de Dados Brutos"),
              textOutput("raw_data_status"), # Status do carregamento dos dados brutos
              fluidRow(
                box(title = "Visualização de Dados Brutos", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("raw_data_table")
                )
              ),
              fluidRow(
                box(title = "Distribuição de Product_Type (Dados Brutos)", status = "info", solidHeader = TRUE, width = 6,
                    plotlyOutput("raw_product_type_plot")
                )
              )
      ),
      
      # Processed Data Tab
      tabItem(tabName = "processed_data",
              h2("Análise de Dados Processados"),
              textOutput("processed_data_status"), # Status do carregamento dos dados processados
              fluidRow(
                box(title = "Visualização de Dados Processados", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("processed_data_table")
                )
              ),
              fluidRow(
                box(title = "Frequência de Tokens (Dados Processados)", status = "info", solidHeader = TRUE, width = 6,
                    plotlyOutput("processed_tokens_plot")
                )
              )
      ),
      
      # Predictions Tab
      tabItem(tabName = "predictions",
              h2("Análise de Previsões"),
              textOutput("predictions_data_status"), # Status do carregamento dos dados de previsão
              fluidRow(
                box(title = "Visualização de Previsões", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("predictions_table")
                )
              ),
              fluidRow(
                box(title = "Distribuição de Sentimentos (Previsões)", status = "info", solidHeader = TRUE, width = 6,
                    plotlyOutput("predictions_sentiment_plot")
                ),
                box(title = "Distribuição de Probabilidades (Previsões)", status = "info", solidHeader = TRUE, width = 6,
                    plotlyOutput("predictions_probability_plot")
                )
              )
      )
    )
  )
)

# Server
server <- function(input, output) {
  
  # Load Data
  raw_data_path <- "../data/raw/Test.csv"
  processed_data_path <- "../data/processed/processed_data.csv"
  predictions_data_path <- "../data/processed/predictions.csv" # Assumindo que o arquivo de previsões estará aqui quando existir

  raw_data_reactive <- reactive({
    data <- tryCatch(read_csv(raw_data_path, show_col_types = FALSE), error = function(e) {
      message("Erro ao carregar dados brutos de '", raw_data_path, "': ", e$message)
      data.frame(Text_ID = character(), Product_Description = character(), Product_Type = character()) # Adjusted columns for Test.csv
    })
    return(data)
  })

  processed_data_reactive <- reactive({
    data <- tryCatch(read_csv(processed_data_path, show_col_types = FALSE), error = function(e) {
      message("Erro ao carregar dados processados de '", processed_data_path, "': ", e$message)
      data.frame(Text_ID = character(), Product_Description = character(), Product_Type = character(), text_clean = character(), text_no_stopwords = character(), text_lemmatized = character(), sentiment = character()) # Adjusted columns for processed_data.csv
    })
    return(data)
  })

  predictions_data_reactive <- reactive({
    data <- tryCatch(read_csv(predictions_data_path, show_col_types = FALSE), error = function(e) {
      message("Erro ao carregar dados de previsão de '", predictions_data_path, "': ", e$message)
      data.frame(id = character(), text = character(), predicted_sentiment = character(), probability = numeric())
    })
    return(data)
  })

  # Adicionar uma verificação para exibir mensagem na UI se os dados não forem carregados
  output$raw_data_status <- renderText({
    if (nrow(raw_data_reactive()) == 0) {
      "Dados Brutos: Não disponível ou erro ao carregar."
    } else {
      "Dados Brutos: Carregados com sucesso."
    }
  })

  output$processed_data_status <- renderText({
    if (nrow(processed_data_reactive()) == 0) {
      "Dados Processados: Não disponível ou erro ao carregar."
    } else {
      "Dados Processados: Carregados com sucesso."
    }
  })

  output$predictions_data_status <- renderText({
    if (nrow(predictions_data_reactive()) == 0) {
      "Dados de Previsão: Não disponível ou erro ao carregar."
    } else {
      "Dados de Previsão: Carregados com sucesso."
    }
  })
  
  # Render Raw Data Table
  output$raw_data_table <- renderDT({
    datatable(raw_data_reactive())
  })
  
  # Render Processed Data Table
  output$processed_data_table <- renderDT({
    datatable(processed_data_reactive())
  })
  
  # Render Predictions Table
  output$predictions_table <- renderDT({
    datatable(predictions_data_reactive())
  })
  
  # Raw Data Sentiment Plot
  output$raw_product_type_plot <- renderPlotly({
    data <- raw_data_reactive()
    if (nrow(data) > 0 && "Product_Type" %in% colnames(data)) {
      data %>% 
        count(Product_Type) %>% 
        plot_ly(x = ~Product_Type, y = ~n, type = "bar", name = "Product Type") %>%
        layout(title = "Distribuição de Product_Type",
               xaxis = list(title = "Product_Type"),
               yaxis = list(title = "Contagem"))
    } else {
      plotly_empty() %>% layout(title = "Dados brutos não disponíveis ou coluna 'Product_Type' ausente")
    }
  })
  
  # Processed Data Tokens Plot (example: top 10 tokens)
  output$processed_tokens_plot <- renderPlotly({
    data <- processed_data_reactive()
    if (nrow(data) > 0 && "text_lemmatized" %in% colnames(data)) {
      tokens_list <- unlist(strsplit(as.character(data$text_lemmatized), " "))
      token_counts <- as.data.frame(table(tokens_list)) %>% 
        arrange(desc(Freq)) %>% 
        head(10)
      
      plot_ly(token_counts, x = ~tokens_list, y = ~Freq, type = "bar", name = "Tokens") %>%
        layout(title = "Top 10 Tokens",
               xaxis = list(title = "Token"),
               yaxis = list(title = "Frequência"))
    } else {
      plotly_empty() %>% layout(title = "Dados processados não disponíveis ou coluna 'text_lemmatized' ausente")
    }
  })
  
  # Predictions Sentiment Plot
  output$predictions_sentiment_plot <- renderPlotly({
    data <- predictions_data_reactive()
    if (nrow(data) > 0 && "predicted_sentiment" %in% colnames(data)) {
      data %>% 
        count(predicted_sentiment) %>% 
        plot_ly(x = ~predicted_sentiment, y = ~n, type = "bar", name = "Sentimento Previsto") %>%
        layout(title = "Distribuição de Sentimentos Previstos",
               xaxis = list(title = "Sentimento Previsto"),
               yaxis = list(title = "Contagem"))
    } else {
      plotly_empty() %>% layout(title = "Dados de previsão não disponíveis ou coluna 'predicted_sentiment' ausente")
    }
  })
  
  # Predictions Probability Plot
  output$predictions_probability_plot <- renderPlotly({
    data <- predictions_data_reactive()
    if (nrow(data) > 0 && "probability" %in% colnames(data)) {
      data %>% 
        plot_ly(x = ~probability, type = "histogram", name = "Probabilidade") %>%
        layout(title = "Distribuição de Probabilidades de Previsão",
               xaxis = list(title = "Probabilidade"),
               yaxis = list(title = "Frequência"))
    } else {
      plotly_empty() %>% layout(title = "Dados de previsão não disponíveis ou coluna 'probability' ausente")
    }
  })
}

# Run the application 
shinyApp(ui = ui, server = server)