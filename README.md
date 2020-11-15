# IMDB Sentiment Analysis
A repository to explore the Doc2Vec algorithm on the IMDB dataset. 

## Data Source 
The dataset can be downloaded from Kaggle [here](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format).
It comes already split into ***Train***, ***Validation*** and ***Test*** sets.

## Analysis Pipeline 

### 1 Clean Raw Text 
The script reads in the raw ***Train***, ***Validation*** and ***Test*** sets and writes out
cleaned datasets. 
It uses **BeautifulSoup** to remove html tags and **spaCy** library to perform the following
tasks:

1) remove stopwords 
2) remove punctuation
3) retains alpha characters only
```
python src/pre-processing.py
```
### 2 Build Doc2Vec Model
The script uses **gensim** library to build a model that represents the entire 
movie review document as a fixed length vector. Initially, the model is parameterized to 
represent each review in 100 dimensions. 

The model is built using the ***Train*** set only. After the Doc2Vec model is trained, 
it will be used to infer the document vectors for the ***Validation***  and ***Test*** sets. 

```
python src/doc2vec.py
```

### 3 Extract Doc2Vec Features 
This script uses the Doc2Vec model produced in step 2 to extract individual document vectors
for the ***Train*** set and infer the individual 
document vectors for the ***Validation*** and ***Test*** portions of the data. The extracted
vectors are then saved as a new dataset where each row is a movie review with 100 features 
learned from the model.   

```
python src/feature_extraction.py
```

# Doc2Vec Model Experimentation 
The sentiment analysis task was carried using multiple `doc2vec` models. The parameters of the 
different models is documented below. 

**Version 1: (PV-DM)** 

This model averages together the paragraph vector along with the 
learned context window word vectors to perform the prediction task of predicting the 
next word in the sequence. 

- max_epochs = 100
- vec_size = 100
- min_count=2
- alpha = 0.025
- dm=1
- window=10

**Version 2: (PV-DM)**

The only difference between this model and the one above is that instead of 
averaging the paragraph vector and context word vectors together before the prediction
task, it concatenates the paragraph vector and the window context word vectors together. 
This model is supposed to improve performance due to the concatentation. However, this
model produces a very large model object (344MB + 50MB)
- max_epochs = 100
- vec_size = 100
- min_count=2
- alpha = 0.025
- dm=1
- window=10
- dm_concat = 1



**Version 3: (PV-DBOW)**

For this model, only the paragraph vectors are trained without the word vectors. 
The network is trained to predict the words occurring in the context window, given
the paragraph vector. 

- max_epochs = 100
- vec_size = 100
- min_count=2
- alpha = 0.025
- dm=0
- window=10