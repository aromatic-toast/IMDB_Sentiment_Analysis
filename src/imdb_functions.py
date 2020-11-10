# Author: Lesley Miller
# Date: 2020/11/01


# Purpose:
# contains helper functions for text processing

import pandas as pd
# text pre-processing
from bs4 import BeautifulSoup

# tokenization
import spacy

# document modelling
from gensim.models.doc2vec import Doc2Vec


def remove_html(text):
    """
    Produces the text with html tags removed and converts to all lower case.

    Arguments
    ---------
    text (pandas.core.series.Series) A series of text documents.

    Returns
    -------
    pandas.core.series.Series A series of text with html tags removed & lower case letters.

    """
    # initialize a list for cleaned text
    cleaned_text = []
    for doc in text:
        ## remove html tags with beautifulsoup
        soup = BeautifulSoup(doc)
        text = soup.get_text().lower()

        # append the text to a new series
        cleaned_text.append(text)

    # convert list to a pandas series
    cleaned_text = pd.Series(cleaned_text)

    return cleaned_text


def clean_text(text):
    """
        Returns a series of cleaned documents with:
        1) spacy stopwords removed
        2) removed punctuation
        3) retains alpha chars only

        Arguments
        ---------
        text (pandas.core.series.Series) A series of text documents.

        Returns
        -------
        (pandas.core.series.Series)
            Produces a series of cleaned up documents.

        """
    # load spacy model
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    # convert each document to a list of tokens of type string
    docs = []
    for doc in nlp.pipe(text):
        doc_tokens = []
        for token in doc:
            # filter for tokens that are NOT stopwords, are NOT punctuation and are alpha chars
            if sum([not token.is_stop, not token.is_punct, token.is_alpha]) == 3:
                doc_tokens.append(token.lemma_)

        # append the filtered tokens to the final document
        docs.append(" ".join(doc_tokens))

    # convert docs to a pandas series
    docs = pd.Series(docs)

    return docs


def extract_doc_vecs(model, vec_size, infer, docs=None):
    """
    Produces a dataframe where each row is the doc2vec vector for each document.

    Parameters
    ----------
    d2v_model (gensim.models.doc2vec.Doc2Vec) A saved doc2vec model object.
    vec_size (int) The number of dimensions used to create the document vectors
    infer    (bool) If True, uses the model to infer the document vector.
    docs     (list) A list where each element of the list is a list of text tokens.

    Returns
    -------
    pandas.core.frame.DataFrame
        A dataframe with each row a doc2vec vector of the text document.

    """
    # initialize a dict to store the doc vectors
    vector_dict = {}
    colnames = []

    # check if vectors should be inferred
    if infer:
        i = 0
        for doc in docs:
            # infer the document vector
            vector_dict[i] = model.infer_vector(doc)
            i += 1
    else:
        for i in range(len(model.docvecs)):
            # build the dict of doc vectors
            vector_dict[i] = model.docvecs[i]

    # create the column names
    for dim in range(vec_size):
        colname = "dim_{0}".format(dim)
        colnames.append(colname)

    # create a dataframe of doc vectors
    vector_df = pd.DataFrame(vector_dict).transpose()

    # set the col names to be number of dimensions of the doc vectors
    vector_df.columns = colnames

    return vector_df

def tokenize(text):
    """
    Converts each document into a list of text tokens.
    Arguments
    ---------
    text (pandas.core.series.Series) A series of text documents. 
    
    Returns
    -------
    list
        A list of list of text tokens.
    """
    # load spacy model
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

    # convert each document into list of text tokens
    docs = []
    for doc in nlp.pipe(text):
        doc = [token.text for token in doc]

        # append the doc to list of documents
        docs.append(doc)

    return docs



