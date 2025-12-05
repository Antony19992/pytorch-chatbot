from fastapi import FastAPI
from pydantic import BaseModel
import torch
import random
import json
import pickle

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

# Carregar modelo salvo com pickle
FILE = "data.pkl"
with open(FILE, "rb") as f:
    data = pickle.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Beijinho"

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "ok", "message": "API do chatbot está funcionando!"}

@app.post("/chat")
def chat(msg: ChatMessage):
    sentence = msg.message
    tokenized = tokenize(sentence)
    X = bag_of_words(tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return {
                    "bot": bot_name,
                    "tag": tag,
                    "confidence": float(prob.item()),
                    "response": random.choice(intent["responses"])
                }
    else:
        return {
            "bot": bot_name,
            "tag": "unknown",
            "confidence": float(prob.item()),
            "response": "Desculpe, não consegui entender. Pode reformular a frase?"
        }
