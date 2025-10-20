# Senti-Pred

Projeto de análise de sentimentos com processamento de linguagem natural e aprendizado de máquina.

## Visão Geral

O Senti-Pred é um projeto de análise de sentimentos que utiliza técnicas de processamento de linguagem natural e aprendizado de máquina para classificar textos de acordo com o sentimento expresso. Este repositório oferece diferentes abordagens para explorar e utilizar o pipeline de análise de sentimentos.

## Opções de Execução

O projeto pode ser executado de três maneiras principais, cada uma com sua própria documentação detalhada:

1.  **Scripts Python Modulares**: Para uma execução mais controlada e modular, ideal para integração em outros sistemas ou automação.
    -   Veja o guia completo em: [README_scripts.md](README_scripts.md)

2.  **Notebook Jupyter**: Para uma exploração interativa e desenvolvimento passo a passo do pipeline completo.
    -   Veja o guia completo em: [README_jupyter.md](README_jupyter.md)

3.  **GCP Vertex AI + Docker**: Para deploy em ambiente de produção utilizando a infraestrutura do Google Cloud Platform e Docker.
    -   Veja o guia completo em: [README_gcp.md](README_gcp.md)

## Estrutura do Projeto

```
senti-pred/
├── README.md
├── README_scripts.md
├── README_jupyter.md
├── README_gcp.md
├── data/
│   ├── raw/ (dados originais)
│   └── processed/ (dados processados)
├── notebooks/
│   └── full_pipeline.ipynb (Pipeline completo)
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

## Contato

Para dúvidas ou sugestões, entre em contato através de pedromoratolahoz@gmail.com.