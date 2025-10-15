# Senti-Pred

Projeto de análise de sentimentos com processamento de linguagem natural e aprendizado de máquina.

## Visão Geral

O Senti-Pred é um projeto de análise de sentimentos que utiliza técnicas de processamento de linguagem natural e aprendizado de máquina para classificar textos de acordo com o sentimento expresso. O projeto segue uma estratégia híbrida, utilizando Jupyter Notebooks para análise exploratória e Azure Machine Learning para treinamento e implantação de modelos.

## Estrutura do Projeto

```
senti-pred/
├── README.md
├── data/
│   ├── raw/ (dados originais)
│   └── processed/ (dados limpos)
├── notebooks/
│   ├── 01_eda.ipynb (Análise exploratória)
│   ├── 02_preprocessing.ipynb (Limpeza)
│   ├── 03_modeling.ipynb (Modelos)
│   └── 04_evaluation.ipynb (Avaliação)
├── src/
│   ├── models/ (código dos modelos)
│   ├── api/ (Django API)
│   └── utils/ (Funções auxiliares)
├── reports/
│   ├── relatorio_tecnico.pdf
│   └── visualizacoes/ (gráficos em R)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/
├── requirements.txt
└── docker-compose.yml (bonus)
```

## Fases do Projeto

O projeto Senti-Pred é dividido em 4 fases principais:

1. **Análise de Dados (EDA)**: Exploração e visualização dos dados para entender padrões e características.
2. **Pré-processamento**: Limpeza e transformação dos dados para prepará-los para modelagem.
3. **Modelagem**: Desenvolvimento e treinamento de modelos de análise de sentimentos.
4. **Avaliação**: Teste e avaliação do desempenho dos modelos desenvolvidos.

## Estratégia Híbrida

| Etapa do Projeto | Plataforma Recomendada | Por que é Estratégica |
|------------------|------------------------|------------------------|
| Análise de Dados (EDA) | Jupyter Notebook (Local) | Permite o uso de R para análise estatística aprofundada (EDA), que é um diferencial único no projeto. |
| Treinamento e Otimização | Azure Machine Learning | Usar o AutoML e o Model Registry no Azure prova habilidades em MLOps de nível empresarial. |
| Deployment do Modelo | Azure ML Endpoints | A melhor prática de AI Engineering é deployar o modelo diretamente na nuvem. O Azure gera a API REST segura que o Django consumirá. |
| Backend API (Intermediário) | Django (Local ou Render/Railway) | Use Django para construir a API que faz a ponte entre o frontend e a API do Azure ML. Isso demonstra arquitetura de microsserviços. |

## Instalação e Configuração

### Pré-requisitos

- Python 3.8+
- Pip (gerenciador de pacotes Python)
- Virtualenv (opcional, mas recomendado)

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/senti-pred.git
   cd senti-pred
   ```

2. Crie e ative um ambiente virtual (opcional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   ```bash
   cp .env.example .env
   # Edite o arquivo .env com suas configurações
   ```

## Uso

### Análise Exploratória e Pré-processamento

Execute os notebooks na ordem:

1. `notebooks/01_eda.ipynb` - Análise exploratória dos dados
2. `notebooks/02_preprocessing.ipynb` - Pré-processamento dos dados

### Treinamento de Modelos

Execute o notebook de modelagem:

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Ou treine o modelo diretamente:

```bash
python src/models/sentiment_model.py
```

### API Django

Para iniciar a API Django:

```bash
cd src/api
python manage.py runserver
```

A API estará disponível em `http://localhost:8000/api/`.

## Testes

Execute os testes unitários:

```bash
pytest tests/unit/
```

Execute os testes de integração:

```bash
pytest tests/integration/
```

Execute os testes de aceitação:

```bash
pytest tests/acceptance/
```

## Containerização

Para executar o projeto em contêineres Docker:

```bash
docker-compose up
```

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça commit das suas alterações (`git commit -am 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.

## Contato

Para dúvidas ou sugestões, entre em contato através de [seu-email@exemplo.com].