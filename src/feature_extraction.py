# Author: Lesley Miller
# Date: 2020/11/10

# this script extracts the doc2vec vectors for the cleaned validation and test sets

# import packages
import sys
sys.path.append("/Users/lesleymi/data_science_tutorials/IMDB_Sentiment_Analysis/src")
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import imdb_functions as imdb

# load d2v model
model = Doc2Vec.load("results/d2v.model")

# load cleaned validation set
valid = pd.read_csv("data/valid_clean.csv")
valid = imdb.tokenize(valid.text)

# load clean test set
test = pd.read_csv("data/test_clean.csv")
test = imdb.tokenize(test.text)

# infer doc vectors for validation set
valid_df = imdb.extract_doc_vecs(model=model, vec_size=100, infer=True, docs=valid)

# infer doc vectors for test set
test_df = imdb.extract_doc_vecs(model=model, vec_size=100, infer=True, docs=test)

# write out the extracted vectors as csvs
valid_df.to_csv("data/valid_d2v.csv")
test_df.to_csv("data/test_d2v.csv")