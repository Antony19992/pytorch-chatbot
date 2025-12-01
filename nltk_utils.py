import numpy as np
import nltk
import unicodedata
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def remove_acentos(texto):
    """
    Remove acentos de um texto usando unicodedata.
    Exemplo: 'Olá Café' -> 'Ola Cafe'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def tokenize(sentence):
    """
    Normaliza a frase (minúsculo + sem acento) e divide em tokens.
    """
    sentence = remove_acentos(sentence.lower())
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Reduz a palavra à sua raiz (stemming).
    Exemplo: 'organizing' -> 'organ'
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Cria um vetor binário (bag of words):
    - 1 se a palavra conhecida aparece na frase
    - 0 caso contrário
    """
    # aplica stemming em cada palavra da frase
    sentence_words = [stem(word) for word in tokenized_sentence]
    # inicializa vetor com zeros
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
