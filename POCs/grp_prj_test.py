import nltk
import gensim
import sklearn
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)
#print(list(newsgroups_train.target_names))

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
nltk.download('wordnet')
print(WordNetLemmatizer().lemmatize('went', pos = 'v'))
import pandas as pd
stemmer = SnowballStemmer("english")
'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result
'''
Preview a document after preprocessing
'''
document_num = 50
doc_sample = 'This disk has failed many times. I would like to get it replaced.'

print("Original document: ")
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print("\n\nTokenized and lemmatized document: ")
print(preprocess(doc_sample))

processed_docs = []

for doc in newsgroups_train.data:
    processed_docs.append(preprocess(doc))
#print(processed_docs[:2])

dictionary = gensim.corpora.Dictionary(processed_docs)
'''
Checking dictionary created
'''
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
document_num = 20
bow_doc_x = bow_corpus[document_num]

#for i in range(len(bow_doc_x)):
    #print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0],
                                                     #dictionary[bow_doc_x[i][0]],
                                                     #bow_doc_x[i][1]))
lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics = 8,
                                   id2word = dictionary,
                                   passes = 10,
                                   workers = 2)
'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
#for idx, topic in lda_model.print_topics(-1):
    #print("Topic: {} \nWords: {}".format(idx, topic ))
    #print("\n")
num = 100
unseen_document = newsgroups_test.data[num]
#print(unseen_document)
# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
