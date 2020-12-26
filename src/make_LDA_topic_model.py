# Author: Lesley Miller
# 2020/12/06

"""
The purpose of this script is to perform LDA on the corpus
and save the model results to disk.
"""

# import packages
import pandas as pd
from time import time

# gensim LDA
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamulticore import LdaMulticore

# load docs
docs = pd.read_parquet("data/train_clean.parquet")
docs = docs.tokenized_docs

# initialize the dictionary
imdb_dictionary = Dictionary(docs)
print("There are {} documents in the dictionary.".format(imdb_dictionary.num_docs))
print("There are {} unique words in the dictionary.".format(imdb_dictionary.num_pos))

# convert each document into a bag of words
"""
This converts the list of tokenized docs into a list of tuples 
for each document.
Where each tuple within a doc consists of a word id and the number of times 
that word occurred in that document.
"""
imdb_corpus = [imdb_dictionary.doc2bow(doc) for doc in docs]

# set model parameters
num_topics = 7
chunk_size = 2000
passes=10
workers = 3
alpha = 'asymmetric'  # the prior belief about each topic's probability

"""
Setting per_word_topics = True will have the model compute a list of topics in 
descending order of most likely topics for each word
"""
start_time = time()
print("Training LDA model with {} topics".format(num_topics))
model = LdaMulticore(corpus=imdb_corpus,
                     id2word=imdb_dictionary,
                     num_topics=num_topics,
                     chunksize=chunk_size,
                     passes=passes,
                     workers=workers,
                     alpha=alpha,
                     per_word_topics=True,
                     random_state=123)
stop_time = time()
print("Time to train the model: {} seconds".format(round(stop_time - start_time)))

# save the model
model.save('results/lda_model1.gensim')