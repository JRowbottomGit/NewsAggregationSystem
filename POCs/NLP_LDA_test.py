import json
import pandas as pd
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from gensim.models import LdaModel


from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

np.random.seed(400)
nltk.download('wordnet')
print(WordNetLemmatizer().lemmatize('went', pos = 'v'))

stemmer = SnowballStemmer("english")

#class NLP_model(object, X, y):
class NLP_model(object):

    def __init__(self):
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

#	def __init__(self, X, y):
		# self.X = X
		# self.y = y
		#self.test_size = 0.1
		#self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(self.test_size)

		# print(f"""The data X,y has been split into:
		# 	X_train {len(self.X_train)}, X_test {len(self.X_test)}, y_train {len(self.y_train)}, y_test {len(self.y_test)}""")
        self.processed_docs = None
        self.dictionary = None
        self.bow_corpus = None

    def split_train_test(self, test_size):
        """
		split the data into training/testing data and labels
		:param test-size: the percentagdictionarye of the data that should be used for testing
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

### these are the preprocessing functions that will go ito the Class news_archive()
    # Tokenize and lemmatize
    # at some point check if spacy does better lemmatization
    def preprocess(self, text):
        '''
        input: story text
            -Tokensizes: story -> sentences -> words
            -removes stop words and words < 3 characters
            -lemmatizes - first person present tense
            -stemming - root word form
        '''
        result=[]
        for token in gensim.utils.simple_preprocess(text) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result

    #Defining pre process functions
    def lemmatize_stemming(self, text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess_story_list(self, news_story_list):
        '''
        loop through news json for a given category and create list of preprocessed stories
        need to decide if saves this list down as json or returns it
        '''
        processed_docs = []
        for doc in news_story_list:
            processed_docs.append(self.preprocess(doc))
        self.processed_docs = processed_docs
        self.dictionary = gensim.corpora.Dictionary(processed_docs)

    def dict_to_BoW(self):
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]
###


class LDA_model(NLP_model):

    def __init__(self):
        super().__init__()

    def train_lda_model(self, bow_corpus, save_name = None):
        #https://radimrehurek.com/gensim/models/ldamulticore.html
        self.model =  gensim.models.LdaMulticore(bow_corpus,
                                           num_topics = 8,
                                           id2word = self.dictionary,
                                           passes = 10,
                                           workers = 2)

        #Save model to disk.
        save_name = "/vol/project/2019/545/g1954505/news-aggregation-system/LDA_models/LDA_model_politics"
        #save_name = "/vol/project/2019/545/g1954505/news-aggregation-system/LDA_models/LDA_model"
        #save_name = datapath("model")
        print("saving here")
        print(save_name)
        self.model.save(save_name)


    def news_topics(self):
        '''
        #given a date, source, topic pick a random story
        #from news_history.json
        returns the (RSS_summary,topics)
        '''
        news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/working_code/news_history.json"
        #need to point this at news_archive
        with open(news_history_path, "r") as f:
            news_history = json.load(f)

        topisised_stories = {}

        for key, value in news_history["20200207"]["BBC"]["politics"].items():
            news_story = value["story"]
            bow_vector = self.dictionary.doc2bow(self.preprocess(news_story))

            topic_string = []
            for index, score in sorted(self.model[bow_vector], key=lambda tup: -1*tup[1]):
                #print("Score: {}\t Topic: {}".format(score, self.model.print_topic(index, 5)))
                topic_string.append("Score: {}\t Topic: {}".format(score, self.model.print_topic(index, 5)))

            topisised_stories[key] =(value["title"],topic_string)

        topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/topics_politics.json"
        #topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/topics.json"

        with open(topics, "w") as f:
            json.dump(topisised_stories, f)

    def print_topics(self):
        #topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/topics.json"
        topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/topics_politics.json"
        with open(topics, "r") as f:
            data = json.load(f)
        json_string = json.dumps(data, indent=4)#, sort_keys=True))
        print(json_string)


if __name__ == "__main__":

    #intitalise the LDA Classifier
    LDA = LDA_model()
    #add the news 20 data as training/test data
    data_home = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive"
    #newsgroups_train = fetch_20newsgroups(data_home = data_home, subset='train', shuffle = True)
    # categories = [
    #                 # 'alt.atheism',
    #                 #  'comp.graphics',
    #                 #  'comp.os.ms-windows.misc',
    #                 #  'comp.sys.ibm.pc.hardware',
    #                 #  'comp.sys.mac.hardware',
    #                 #  'comp.windows.x',
    #                 #  'misc.forsale',
    #                 #  'rec.autos',
    #                 #  'rec.motorcycles',
    #                 #  'rec.sport.baseball',
    #                 #  'rec.sport.hockey',
    #                 #  'sci.crypt',
    #                 #  'sci.electronics',
    #                 #  'sci.med',
    #                 #  'sci.space',
    #                 #  'soc.religion.christian',
    #                 #  'talk.politics.guns',
    #                 #  'talk.politics.mideast',
    #                  'talk.politics.misc'#,
    #                  # 'talk.religion.misc'
    #                  ]
    #newsgroups_train_politics = fetch_20newsgroups(data_home = data_home, subset='train', categories=categories, shuffle = True)
    newsgroups_train_politics = fetch_20newsgroups(data_home = data_home, subset='train', shuffle = True)
    newsgroups_test = fetch_20newsgroups(data_home = data_home, subset='test', shuffle = True)


# Preprocess and save news data as gensim.corpora.Dictionary attribute of LDA model
    # LDA.preprocess_story_list(newsgroups_train.data)
    LDA.preprocess_story_list(newsgroups_train_politics.data)
    LDA.dict_to_BoW()

    #LDA.dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

# train the LDA Classifier
    #LDA.train_lda_model(LDA.bow_corpus)

# Load a pretrained model from disk.
    LDA_file = "/vol/project/2019/545/g1954505/news-aggregation-system/LDA_models/LDA_model"
    #LDA_file = "/vol/project/2019/545/g1954505/news-aggregation-system/LDA_models/LDA_model_politics"
    LDA.model = LdaModel.load(LDA_file)

    #test point
    # num = 100
    # unseen_document = newsgroups_test.data[num]
    # #print(unseen_document)
    # # Data preprocessing step for the unseen document
    # bow_vector = LDA.dictionary.doc2bow(LDA.preprocess(unseen_document))
    #
    # for index, score in sorted(LDA.model[bow_vector], key=lambda tup: -1*tup[1]):
    #     print("Score: {}\t Topic: {}".format(score, LDA.model.print_topic(index, 5)))

    LDA.news_topics()
    LDA.print_topics()
