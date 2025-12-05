import random
import json
import re
import joblib
import unicodedata

from nltk_utils import bag_of_words, tokenize

# Função para remover acentos
def strip_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

# Função de normalização
def normalize_text(text):
    text = strip_accents(text.lower())
    text = re.sub(r'(.)\1+', r'\1', text)   # reduz repetições de letras
    text = re.sub(r'[^\w\s]', '', text)     # remove símbolos/emoji
    return text

# Carregar intents
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Carregar modelo treinado
FILE = "data.pkl"
data = joblib.load(FILE)

model = data["model"]
all_words = data["all_words"]
tags = data["tags"]

bot_name = "Beijinho"
print("Olá, em que posso ajudar hoje? (escreva 'quit' para sair)")

while True:
    sentence = input("Você: ")
    if sentence.lower() == "quit":
        break

    # Normalizar entrada
    normalized = normalize_text(sentence)

    # Tokenizar e bag-of-words
    tokens = tokenize(normalized)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, -1)

    # Previsão com modelo
    predicted = model.predict(X)[0]
    probas = model.predict_proba(X)[0]
    prob = max(probas)

    tag = tags[predicted]

    # Fallback usando intents.json com pontuação
    if prob < 0.75:
        scores = {}
        for intent in intents['intents']:
            score = 0
            for pattern in intent['patterns']:
                norm_pattern = normalize_text(pattern)
                # match exato ou parcial
                if norm_pattern in normalized:
                    score += len(norm_pattern)  # mais pontos para padrões mais longos
            if score > 0:
                scores[intent['tag']] = score

        if scores:
            # pega a intent com maior pontuação
            tag = max(scores, key=scores.get)
            prob = 1.0

    # Responder
    found_response = False
    if prob > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                found_response = True
                break

    if not found_response:
        print(f"{bot_name}: Desculpe, não consegui entender, mas caso precise fazer um pedido, saber sobre valores ou até outros assuntos é só me dizer...")
