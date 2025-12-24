# Usar uma imagem base oficial do Python
FROM python:3.9-slim

# Definir o diretório de trabalho no container
# Usamos /app para simular a raiz do projeto
WORKDIR /app

# Copiar os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o conteúdo do projeto para o container
# Isso é necessário para que o dashboard acesse ../reports/metrics e ../reports/visualizacoes
COPY . .

# Expor a porta padrão do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação
# Executamos a partir da raiz (/app) apontando para o arquivo do dashboard
CMD ["streamlit", "run", "streamlit_dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
