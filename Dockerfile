# -------------------------------------------
# 1) Imagem base (Python 3.11 para compatibilidade com torch 2.0.1)
# -------------------------------------------
    FROM python:3.11-slim

    # Evitar prompts e mensagens desnecessárias
    ENV DEBIAN_FRONTEND=noninteractive
    
    # -------------------------------------------
    # 2) Instalar dependências do sistema
    # -------------------------------------------
    RUN apt-get update && apt-get install -y \
        build-essential \
        gcc \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
    # -------------------------------------------
    # 3) Diretório da aplicação
    # -------------------------------------------
    WORKDIR /app
    
    # -------------------------------------------
    # 4) Copiar somente requirements.txt primeiro (cache Docker)
    # -------------------------------------------
    COPY requirements.txt .
    
    # -------------------------------------------
    # 5) Atualizar pip e instalar dependências Python
    # -------------------------------------------
    RUN pip install --upgrade pip
    RUN pip install --no-cache-dir -r requirements.txt
    
    # -------------------------------------------
    # 6) Copiar o restante do projeto
    # -------------------------------------------
    COPY . .
    
    # -------------------------------------------
    # 7) Baixar recursos do NLTK
    # -------------------------------------------
    RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    
    # -------------------------------------------
    # 8) Rodar o treinamento do modelo
    # Isso vai gerar o arquivo data.pth dentro da imagem
    # -------------------------------------------
    RUN python train.py
    
    # -------------------------------------------
    # 9) Expor porta (caso transforme em API depois)
    # -------------------------------------------
    EXPOSE 5000
    
    # -------------------------------------------
    # 10) Comando padrão para iniciar o chatbot
    # -------------------------------------------
    CMD ["python", "chat.py"]
    