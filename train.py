import numpy as np
import random
import json
import joblib

from sklearn.naive_bayes import MultinomialNB
from nltk_utils import bag_of_words, tokenize, stem

# Carregar intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Preparar dados
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Criar dataset
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Treinar modelo com Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Salvar modelo e metadados
data = {
    "model": model,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pkl"
joblib.dump(data, FILE)

print(f"Treinamento completo. Modelo salvo em {FILE}")
