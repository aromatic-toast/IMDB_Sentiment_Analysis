# Author: Lesley Miller
# Date: 2020/11/10

# this script extracts the doc2vec vectors for the cleaned validation and test sets

# import packages
import sys
sys.path.append("/Users/lesleymi/data_science_tutorials/IMDB_Sentiment_Analysis/src")
import imdb_functions as imdb
import pandas as pd
from gensim.models.doc2vec import Doc2Vec


# load d2v model
model = Doc2Vec.load("results/d2v_X_train_clean.model")

# extract the train set vectors
print("extracting train set vectors...")
train_df = imdb.extract_doc_vecs(model=model, vec_size=100, infer=False)

# load cleaned validation set
print("loading and tokenizing validation set...")
valid = pd.read_csv("data/valid_clean.csv")
valid = imdb.tokenize(valid.text)

# load clean test set
print("loading and tokenizing test set...")
test = pd.read_csv("data/test_clean.csv")
test = imdb.tokenize(test.text)

# infer doc vectors for validation set
print("inferring validation vectors...")
valid_df = imdb.extract_doc_vecs(model=model, vec_size=100, infer=True, docs=valid)

# infer doc vectors for test set
print("inferring test vectors")
test_df = imdb.extract_doc_vecs(model=model, vec_size=100, infer=True, docs=test)

# write out the extracted vectors as csv
print('writing out vectorized datasets...')
train_df.to_csv("data/train_d2v.csv")
valid_df.to_csv("data/valid_d2v.csv")
test_df.to_csv("data/test_d2v.csv")
