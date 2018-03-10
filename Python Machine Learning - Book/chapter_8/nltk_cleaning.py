import numpy as np

sample_sentence = 'runners like running and thus they run'
sample_sentence_stop = 'a runner likes running and runs a lot'

# 1. Tokenize documents
def tokenizer(text):
    return text.split()
print(tokenizer(sample_sentence))

# 2. Apply Porter stemmer algorithm
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter(sample_sentence)) # notice the behavior of 'thus'!

# 3. Remove 127 common stop-words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter(sample_sentence_stop)[-10:] if w not in stop])