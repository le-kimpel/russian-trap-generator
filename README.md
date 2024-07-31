# russian-trap-generator
Silly side project where I play around with generating Russian rap-song text. Studying a language is more fun when you play around with it in its foulest, least-grammatical, most-inventive forms.

I use a tensorflow (keras) LSTM (Long-Short Term Memory) RNN to generate about 100 words of Russian-language text (given some seed text), which is pulled from a large dataset of complete rap song lyrics. 
Disclaimer: the dataset also contains English-language words (and probably words from other languages as well.) I didn't filter those out - instead, having never played around with mixed-language datasets, I kept the non-Russian tokens. We'll see what that yields, probably something funny.

### Link to the Kaggle dataset: 

https://www.kaggle.com/datasets/anastasiadrozhzhina/ru-songs-1970-2023?resource=download

### Current state of things

* Goal is to spit something -- ANYTHING -- out, grammatical or not (in the sense of coherence, not linguistic inventiveness.)
* Currently VM has too little RAM so I need to give it more juice at some point, which should let me train this model without the Linux kernel nuking the process.
