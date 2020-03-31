import spacy
from collections import Counter
import json
import en_core_web_sm

#python3 -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load('en_core_web_lg')
#https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
#https://spacy.io/usage/linguistic-features #entity-types
#https://spacy.io/api/annotation#named-entities
#https://spacy.io/usage/linguistic-features
#https://stackoverflow.com/questions/37253326/how-to-find-the-most-common-words-using-spacy

def tag_text(text):
    # Load the large English NLP model above
    # Parse the text with spaCy. This runs the entire pipeline.
    doc = nlp(text)
    # 'doc' now contains a parsed version of text. We can use it to do anything we want!
    # For example, this will print out all the named entities that were detected:
    for entity in doc.ents:
        print(f"{entity.text} ({entity.label_})")

def news_NER(date = "20200207"):
    #################THIS IS IN THE WRONG FOLDER
    #news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/news_history.json"
    #news_history_path = "./news_archive/news_history.json"
    news_history_path = "../news_archive/news_history.json"
    #need to point this at news_archive
    with open(news_history_path, "r") as f:
        news_history = json.load(f)
    #create list of news stories
    #story_list = [story_object["story"] for story_object in news_history["20200207"]["BBC"]["politics"].values()]
    #can also loop over dates
    #story_list = [story_object["story"] for story_object in news_history[date]["BBC"]["politics"].values() for date in news_history.keys()]
    #loop over categories
    categories = ['politics', 'business', 'science', 'technology', 'sport', 'entertainment']
    temp_category_ngrams = {}
    for category in categories:
        print(f"{category}\n")
        story_list = [story_object["story"] for story_object in news_history[date]["BBC"][category].values()]

        nouns_pnouns = []
        for doc in nlp.pipe(story_list, disable=["parser"]): #"tagger",
            #nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"] #doc.ents
            #PROPNs = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "PROPN"] #doc.ents

            # if token.text != token.lemma_:
            #     print('Original : %s, New: %s' % (token.text, token.lemma_))

            # nouns_pnouns.append([token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN" or token.pos_ == "PROPN"]) #doc.ents
            nouns_pnouns.append([token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN" or token.pos_ == "PROPN"]) #doc.ents

        #create a list of words, bigrams and trigrams
        total_nouns = []
        singles = []
        [[(singles.append(word),total_nouns.append(word)) for word in doc_word_list] for doc_word_list in nouns_pnouns]
        bigrams = []
        [[(bigrams.append(doc_word_list[i] + " " + doc_word_list[i+1]),total_nouns.append(doc_word_list[i] + " " + doc_word_list[i+1])) for i in range(len(doc_word_list)-1)] for doc_word_list in nouns_pnouns]
        trigrams = []
        [[(trigrams.append(doc_word_list[i] + " " + doc_word_list[i+1] + " " + doc_word_list[i+2]),total_nouns.append(doc_word_list[i] + " " + doc_word_list[i+1] + " " + doc_word_list[i+2])) for i in range(len(doc_word_list)-2)] for doc_word_list in nouns_pnouns]

        print(f"length total_nouns {len(total_nouns)}\n")
        n = 20
        #singles
        single_noun_freq = Counter(singles)
        single_common_nouns =single_noun_freq.most_common(n)
        print(f"Single common_nouns {single_common_nouns}\n")
        #bigrams
        bigram_noun_freq = Counter(bigrams)
        bigram_common_nouns = bigram_noun_freq.most_common(n)
        print(f"Bigram common_nouns {bigram_common_nouns}\n")
        #trigrams
        trigram_noun_freq = Counter(trigrams)
        trigram_common_nouns = trigram_noun_freq.most_common(n)
        print(f"Trigram common_nouns {trigram_common_nouns}\n")
        #total
        noun_freq = Counter(total_nouns)
        common_nouns = noun_freq.most_common(n)
        print(f"Total common_nouns {common_nouns}\n")
        #dedupe
        bi_less_tri = []
#            [[bi_less_tri.append(bigram) for bigram in bigram_common_nouns if bigram not in trigram] for trigram in trigram_common_nouns]
        for bigram in bigram_common_nouns:
            bi_in_tri = 0
            for trigram in trigram_common_nouns:
                if bigram[0] in trigram[0]:
                    bi_in_tri += 1
            if bi_in_tri == 0:
                bi_less_tri.append(bigram)
        single_less_bi_tri = []
        # [single_less_bi_tri.append(word) for word in bigram_common_nouns if word not in trigram_common_nouns and word not in bigram_common_nouns]
        for single in single_common_nouns:
            single_in_bi_tri = 0
            for bigram in bigram_common_nouns:
                for trigram in trigram_common_nouns:
                    if single[0] in bigram[0]:
                        single_in_bi_tri += 1
            if single_in_bi_tri == 0:
                single_less_bi_tri.append(single)
        print(f"length bi_less_tri {len(bi_less_tri)}")
        print(f"Total bi_less_tri {bi_less_tri}\n")
        print(f"length single_less_bi_tri {len(single_less_bi_tri)}")
        print(f"Total single_less_bi_tri {single_less_bi_tri}\n")
        #concat~ trigrams + bi_less_tri + single_less_bi_tri
        conceptNet_input = trigram_common_nouns + bi_less_tri + single_less_bi_tri
        print(f"conceptNet_input {len(conceptNet_input)}")
        print(f"conceptNet_input {conceptNet_input}\n")
        #common_nouns_dict = {category+"single":single_common_nouns, category+"bigrams":bigram_common_nouns, category+"trigrams" : trigram_common_nouns}
        #save top entities to JSON
        #NER_topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/NER_ngram_20200207_all.json"
    # NER_topics = "../news_archive/NER_ngram_20200207_all.json"
    # NER_topics = f"../news_archive/NER_ngram_{date}_all.json"
    # with open(NER_topics, "w") as f:
    #     #json.dump(common_nouns_dict, f)
    #     json.dump(conceptNet_input, f)
        temp_category_ngrams[category] = conceptNet_input
    ngrams_store = f"../news_archive/NER_ngrams_store.json"
    with open(ngrams_store, 'r') as f:
        temp_ngrams_hist = json.load(f)

    temp_ngrams_hist[date] = temp_category_ngrams
    with open(ngrams_store, 'w') as f:
        json.dump(temp_ngrams_hist,f)

def tag_texts(texts):
    for doc in nlp.pipe(texts, disable=["parser"]): #"tagger",
        #nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"] #doc.ents
        #PROPNs = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "PROPN"] #doc.ents
        nouns_pnouns.append([token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN" or token.pos_ == "PROPN"]) #doc.ents

if __name__ == "__main__":
    # temp_ngrams_hist = {}
    # ngrams_store = f"../news_archive/NER_ngrams_store.json"
    # with open(ngrams_store, 'w') as f:
    #     json.dump(temp_ngrams_hist,f)

    news_history_path = "../news_archive/news_history.json"
    with open(news_history_path, "r") as f:
        news_history = json.load(f)

    for date in news_history.keys():
        news_NER(date)
    # news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/working_code/news_history.json"
    # with open(news_history_path, "r") as f:
    #     news_history = json.load(f)
    # print(news_history["20200207"]["BBC"].keys())
    # The text we want to examine
    # texts = [
    #     "Net income was $9.4 million compared to the prior year of $2.7 million.",
    #     "Revenue exceeded twelve billion dollars, with a loss of $1b.","""London is the capital and most populous city of England and
    #     the United Kingdom.  Standing on the River Thames in the south east
    #     of the island of Great Britain, London has been a major settlement
    #     for two millennia. It was founded by the Romans, who named it Londinium.
    #     """,]
    # text = """London is the capital and most populous city of England and
    # the United Kingdom.  Standing on the River Thames in the south east
    # of the island of Great Britain, London has been a major settlement
    # for two millennia. It was founded by the Romans, who named it Londinium.
    # """
    #tag_text(text)
    #tag_texts(texts)
    # news_NER()
