# Imagem base leve com Python 3.11
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copiar requirements.txt primeiro
COPY requirements.txt .

# Instalar dependências Python (CPU-only)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar o restante do projeto
COPY . .

# Baixar recursos do NLTK
RUN python -c "import nltk; nltk.download('punkt')"

# Treinar modelo (gera data.pkl)
RUN python train.py

# Expor porta da API
EXPOSE 5000

# Iniciar FastAPI com Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
