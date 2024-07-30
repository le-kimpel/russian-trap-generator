import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding

'''
model.py

Trains an LSTM to spit out various russian-language song lyrics. 
'''
def generate_model(vocab_size, seq_len):
    '''
    Generates an LSTM with an Embedding layer, two LSTM layers, and two dense layers. This is the default model using the tutorial provided by:

    https://medium.com/@govardhanspace/text-generation-using-lstm-with-keras-in-python-ba5e492bedfa
    '''
    model = Sequential()
    model.add(Embedding(vocab_size, 25, input_length = seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150), activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

with open ("out.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
print(raw_text[0:200])
vocab = list(set(raw_text.split(' ')))

print("Dataset has " + str(len(vocab)) + " unique words")

train_len = 25+1
text_sequences = []
for i in range(train_len, len(vocab)):
    seq = vocab[i-train_len:i]
    text_sequences.append(seq)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

