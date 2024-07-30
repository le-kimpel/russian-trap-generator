import numpy
import tensorflow as tf
import re

'''
model.py

Trains an LSTM to spit out various russian-language song lyrics. 
'''

with open ("out.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
print(raw_text[0:200])
vocab = list(set(raw_text.split(' ')))

print("Dataset has " + str(len(vocab)) + " unique words")

ids_from_words = tf.keras.layers.StringLookup(vocabulary=list(vocab),mask_token=None)

words_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_words.get_vocabulary(), invert=True, mask_token=None)

#text_from_ids = tf.strings.reduce_join(chars_from_ids, axis=-1).numpy()

all_ids = ids_from_words(tf.strings.unicode_split(raw_text, 'UTF-8'))

i#ds_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
