# Usa imagem leve do Python
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia dependências
COPY requirements.txt .

# Instala dependências
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Baixa dados do NLTK
RUN python -c "import nltk; nltk.download('punkt')"

# Roda o treinamento (gera data.pkl)
RUN python train.py

# Expõe a porta usada pelo Uvicorn
EXPOSE 5000

# Inicia a API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
