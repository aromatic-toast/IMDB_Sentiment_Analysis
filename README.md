# IMDB Sentiment Analysis
A repository to explore the Doc2Vec algorithm on the IMDB dataset. 

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

