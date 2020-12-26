# Author: Lesley Miller
# Date: 2020/11/01

# this script performs all the text cleaning steps on each document
# and writes out a new dataset where each row in the dataframe is cleaned text.


# load packages
import sys
sys.path.append("/Users/lesleymi/data_science_tutorials/IMDB_Sentiment_Analysis/src")
import pandas as pd
import imdb_functions as imdb


# load the data
train = pd.read_csv("data/Train.csv")
valid = pd.read_csv("data/Valid.csv")
test = pd.read_csv("data/Test.csv")
y_test = test.label

#########------------TRAIN SET------------##########
# pre-process the train text
print("Removing html on train...")
train_no_html = imdb.remove_html(train.text)

# clean the train set
print("cleaning train...")
train_clean = imdb.clean_text(train_no_html)

# create clean train df
train_clean_dict = {'text':train_clean, 'label': train.label}
train_clean_df = pd.DataFrame(train_clean_dict)

#########------------VALIDATION SET------------##########
# pre-process the validation text
print("Removing html on validation...")
valid_no_html = imdb.remove_html(valid.text)

# clean the validation set
print("cleaning validation...")
valid_clean = imdb.clean_text(valid_no_html)

# create clean validation df
valid_clean_dict = {'text':valid_clean, 'label': valid.label}
valid_clean_df = pd.DataFrame(valid_clean_dict)


#########------------TEST SET------------##########33
# pre-process the test text
print("Removing html on test...")
test_no_html = imdb.remove_html(test.text)

# clean the test set
print("cleaning test...")
test_clean = imdb.clean_text(valid_no_html)

# create clean test df
test_clean_dict = {'text': test_clean, 'label': test.label}
test_clean_df = pd.DataFrame(test_clean_dict)


print("writing out cleaned train, validation & test sets to csv")
train_clean_df.to_csv('data/train_clean.csv', index=False)
valid_clean_df.to_csv('data/valid_clean.csv', index=False)
test_clean_df.to_csv('data/test_clean.csv', index=False)










































































































































































































































