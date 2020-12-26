# Author: Lesley Miller
# Date: 2020/11/28

"""
The purpose of this script is to add the tokenized version of each
document to the raw and processed training sets in parquet format. This script
only needs to be run once, and is intended to save time in downstream applications
that require the text to be tokenized.
"""

# import packages
import pandas as pd
import sys
sys.path.append("/Users/lesleymi/data_science_tutorials/IMDB_Sentiment_Analysis/src")
# custom functions
import imdb_functions as imdb

###### LOAD ############
# load raw/processed train sets
print("Loading data...")
train = pd.read_csv("data/Train.csv")
train_clean = pd.read_csv("data/train_clean.csv").drop(labels='Unnamed: 0', axis=1)

########## TOKENIZE ##########
# convert documents into list of tokens
print("Tokenizing raw docs...")
raw_docs = imdb.tokenize(text=train.text)
# convert clean documents into list of tokens
print("Tokenizing clean docs...")
clean_docs = imdb.tokenize(text=train_clean.text)


######## ADD TOKENIZED DOCS #########
print("Adding tokenized docs to train sets...")
# add tokenized docs to train df
train['tokenized_docs'] = raw_docs

# add tokenized docs to train df
train_clean['tokenized_docs'] = clean_docs

###### EXPORT ##########
print("Exporting train sets...")
train.to_parquet('data/Train.parquet', index=False)
train_clean.to_parquet('data/train_clean.parquet', index=False)