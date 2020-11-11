# Author: Lesley Miller
# Date: 2020/11/11

# this script builds a doc2vec model for each document and writes out
# a saved model object for later use

# load packages
import pandas as pd
import sys
sys.path.append("/Users/lesleymi/data_science_tutorials/IMDB_Sentiment_Analysis/src")
import imdb_functions as imdb

# text modelling
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

# read in clean training text
X_train = pd.read_csv("data/train_clean.csv").text

# tokenize the text into a list of lists
print("Tokenizing training text...")
X_train = imdb.tokenize(X_train)

# build list of Tagged Documents
print("Building list of tagged documents...")
tagged_docs = [TaggedDocument(words=doc, tags=[tag]) for tag, doc in enumerate(X_train)]

# set number of processing cores
cores = multiprocessing.cpu_count()

# set model params
max_epochs = 100
vec_size = 100
min_count=2
alpha = 0.025
dm=1
window=10

# initialize the model
model = Doc2Vec(vector_size=vec_size,
               min_count=min_count,
               dm=dm,
               epochs=max_epochs,
               window=window,
               workers=cores)

# build the vocobulary
print("Building model vocabulary...")
model.build_vocab(tagged_docs)

# train the model
print("Training Doc2Vec model...")
model.train(documents=tagged_docs,
            total_examples=model.corpus_count,
            epochs=model.epochs)

# save the model
print("Saving d2v model to disk...")
model.save("results/d2v_X_train_clean.model")



