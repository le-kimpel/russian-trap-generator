import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
from tensorflow.keras.utils import to_categorical

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
    model.add(Dense(150), activation='relu')
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def generate_text(model, tokenizer, seq_len, seed_text, num_words=100):
    '''
     Generate a paragraph of about 100 words (default) by individually predicting the occurance of each word using the model, and appending to a string.
    '''
    out_text = []
    in_text = seed_text
    for i in range(num_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        predictX = model.predict(pad_encoded, verbose=0)[0]
        pred_word_ind = np.argmax(predictX)
        pred_word = tokenizer.index_word[pred_word_ind]
        in_text += ' ' + pred_word
        out_text.append(pred_word)
    return out_text

if __name__=="__main__":
    
    with open ("out.txt", "r", encoding = "utf-8") as f:
        raw_text = f.read()
    print(raw_text[0:200])
    tokens = []
    clean_text = raw_text.lower()
    clean_text = clean_text.split(' ')
    for word in clean_text:
        if word != '':
            tokens.append(word)
    # I had to limit the size of this dataset; my VM currently cannot handle something this big.
    tokens = tokens[(len(tokens)-400000):len(tokens)]
    print("Dataset has " + str(len(list(set(tokens)))) + " unique words")
    print("Setting up sequences...")
    train_len = 25+1
    text_sequences = []
    for i in range(train_len, len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)

    print("Commencing tokenization...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    vocab_size = len(tokenizer.word_counts)
    sequences = tokenizer.texts_to_sequences(text_sequences)
    sequences = np.array(sequences)
    
    # split vocab into train/test set
    X = sequences[:, :-1]
    y = sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size+1)

    print("Training.")
    # train model
    model = generate_model(vocab_size, seq_len)
    model.fit(X,y, batch_size=128, epochs=300, verbose=1)

    seed_text = ''.join(text_sequences[0])
    print("SEED TEXT: \n " + seed_text)
    s = generate_text(model, tokenizer, seq_len, seed_text = seed_text)
    print(s)
