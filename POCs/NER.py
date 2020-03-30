import spacy
from collections import Counter
import json

nlp = spacy.load('en_core_web_lg')
#https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e
#https://spacy.io/usage/linguistic-features#entity-types
#https://spacy.io/api/annotation#named-entities
#https://spacy.io/usage/linguistic-features
#https://stackoverflow.com/questions/37253326/how-to-find-the-most-common-words-using-spacy

def tag_text(text):
    # Load the large English NLP model

    # Parse the text with spaCy. This runs the entire pipeline.
    doc = nlp(text)

    # 'doc' now contains a parsed version of text. We can use it to do anything we want!
    # For example, this will print out all the named entities that were detected:
    for entity in doc.ents:
        print(f"{entity.text} ({entity.label_})")

def news_NER():
        #################THIS IS IN THE WRONG FOLDER
        news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/news_history.json"
        #need to point this at news_archive
        with open(news_history_path, "r") as f:
            news_history = json.load(f)

        #create list of news stories
        #story_list = [story_object["story"] for story_object in news_history["20200207"]["BBC"]["politics"].values()]

        #can also loop over dates
        #story_list = [story_object["story"] for story_object in news_history[date]["BBC"]["politics"].values() for date in news_history.keys()]

        #loop over categories
        categories = ['politics', 'business', 'science', 'technology', 'sport', 'entertainment']
        for category in categories:

            story_list = [story_object["story"] for story_object in news_history["20200207"]["BBC"][category].values()]

            nouns_pnouns = []
            for doc in nlp.pipe(story_list, disable=["parser"]): #"tagger",
                #nouns = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"] #doc.ents
                #PROPNs = [token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "PROPN"] #doc.ents
                nouns_pnouns.append([token.text for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN" or token.pos_ == "PROPN"]) #doc.ents

            #create a list of words, bigrams and trigrams
            total_nouns = []
            singles = []
            [[(singles.append(word),total_nouns.append(word)) for word in doc_word_list] for doc_word_list in nouns_pnouns]
            bigrams = []
            [[(bigrams.append(doc_word_list[i] + " " + doc_word_list[i+1]),total_nouns.append(doc_word_list[i] + " " + doc_word_list[i+1])) for i in range(len(doc_word_list)-1)] for doc_word_list in nouns_pnouns]
            trigrams = []
            [[(trigrams.append(doc_word_list[i] + " " + doc_word_list[i+1] + " " + doc_word_list[i+2]),total_nouns.append(doc_word_list[i] + " " + doc_word_list[i+1] + " " + doc_word_list[i+2])) for i in range(len(doc_word_list)-2)] for doc_word_list in nouns_pnouns]

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
            print(f"length total_nouns {len(total_nouns)}")
            print(f"Total common_nouns {common_nouns}")

            #dedupe
            # bi_less_tri = []
            # [bi_less_tri.append(word) if word not in trigram_common_nouns for word in bigram_common_nouns]
            # single_less_bi_tri = []
            # [single_less_bi_tri.append(word) if word not in trigram_common_nouns and word not in bigram_common_nouns for word in bigram_common_nouns]


            common_nouns_dict = {category+"single":single_common_nouns, category+"bigrams":bigram_common_nouns, category+"trigrams" : trigram_common_nouns}
            #save top entities to JSON
            NER_topics = "/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/NER_ngram_20200207_all.json"


            with open(NER_topics, "w") as f:
                json.dump(common_nouns_dict, f)

def tag_texts(texts):
    for doc in nlp.pipe(texts, disable=["parser"]): #"tagger",

if __name__ == "__main__":

    # news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/working_code/news_history.json"
    # with open(news_history_path, "r") as f:
    #     news_history = json.load(f)
    # print(news_history["20200207"]["BBC"].keys())

    # The text we want to examine
    texts = [
        "Net income was $9.4 million compared to the prior year of $2.7 million.",
        "Revenue exceeded twelve billion dollars, with a loss of $1b.","""London is the capital and most populous city of England and
        the United Kingdom.  Standing on the River Thames in the south east
        of the island of Great Britain, London has been a major settlement
        for two millennia. It was founded by the Romans, who named it Londinium.
        """,
    ]

    text = """London is the capital and most populous city of England and
    the United Kingdom.  Standing on the River Thames in the south east
    of the island of Great Britain, London has been a major settlement
    for two millennia. It was founded by the Romans, who named it Londinium.
    """
    #tag_text(text)
    #tag_texts(texts)
    news_NER()
