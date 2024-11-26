import json
import pickle
import numpy as np
import random
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and load model/data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentence into a bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict intent
def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Function to get a response based on the predicted intent
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Main function to generate chatbot response
def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, intents)
    return res

# Test the chatbot
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    print("Bot:", chatbot_response(message))
