import random
import json
import re
import pickle
import unicodedata
import torch

from model import NeuralNet
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

# Carregar modelo treinado (pickle)
FILE = "data.pkl"
with open(FILE, "rb") as f:
    data = pickle.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

device = torch.device("cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

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
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Previsão com modelo
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    # Fallback usando intents.json com pontuação
    if prob < 0.75:
        scores = {}
        for intent in intents['intents']:
            score = 0
            for pattern in intent['patterns']:
                norm_pattern = normalize_text(pattern)
                if norm_pattern in normalized:
                    score += len(norm_pattern)
            if score > 0:
                scores[intent['tag']] = score

        if scores:
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
