#https://bigl.es/friday-fun-get-the-news-with-rss-and-gui-zero/
#https://sourceforge.net/projects/xming/ for windows 10 ubuntu gui
import feedparser
import json
import working_code.URL_RSS
import working_code.GetFullText
# import time
import datetime

'''
Creates RSS_data dictionary of the following form:
    {"BBC": {
                "politics" :    {
                        "story_id1" :   {   "link" :
                                            "title" :
                                            "published" :
                                            "summary" :
                                            "story" :
                                        },
                        "story_id2" :   {   "link" :
                                            "title" :
                                            "published" :
                                            "summary" :
                                            "story" :
                                        }
                                }
                "business" :    {...
                                ...}
    "Guardian":{...
    }
'''

# times = {}

def news_today_dic():
    RSS_data = {'BBC':{}}

    for key,value in URL_RSS.URL_RSS_dict['BBC'].items():
        news_rss = feedparser.parse(value[1])
        RSS_data['BBC'][key] = {}
        numCatStories = len(news_rss['entries'])
        # start = time.time()
        for i in range(numCatStories):                         #####################################change this to the length of each topic
            try:
                url = news_rss['entries'][i]['link']
                sttrr = GetFullText.GetFullTextForBBC(url)
            except:
                sttrr = GetFullText.GetFullTextForBBCSimple(url)
            id = news_rss['entries'][i]['id']
            RSS_data['BBC'][key][id] = {}
            url = news_rss['entries'][i]['link']
            print(i,numCatStories,"B")
            RSS_data['BBC'][key][id]['link'] = url
            RSS_data['BBC'][key][id]['title'] = news_rss['entries'][i]['title']
            RSS_data['BBC'][key][id]['published'] = news_rss['entries'][i]['published']
            RSS_data['BBC'][key][id]['summary'] = news_rss['entries'][i]['summary']
            RSS_data['BBC'][key][id]['story'] = sttrr

        # test_time = end - start
        # end = time.time()
        # times["BBC"+key] = (numCatStories,test_time)
        print(numCatStories,"B")
    print("BBC_end")


    RSS_data['theguardian'] = {}

    for key,value in URL_RSS.URL_RSS_dict['theguardian'].items():
        news_rss = feedparser.parse(value[1])
        RSS_data['theguardian'][key] = {}
        numCatStories = len(news_rss['entries'])
        # start = time.time()
        for i in range(numCatStories):                         #####################################change this to the length of each topic
            id = news_rss['entries'][i]['id']
            RSS_data['theguardian'][key][id] = {}
            url = news_rss['entries'][i]['link']
            RSS_data['theguardian'][key][id]['link'] = url
            RSS_data['theguardian'][key][id]['title'] = news_rss['entries'][i]['title']
            RSS_data['theguardian'][key][id]['published'] = news_rss['entries'][i]['published']
            RSS_data['theguardian'][key][id]['summary'] = news_rss['entries'][i]['summary']
            RSS_data['theguardian'][key][id]['story'] = GetFullText.GetFullTextForGuardian(url)
            # print(i,numCatStories)
        # test_time = end - start
        # end = time.time()
        # times["theguardian"+key] = (numCatStories,test_time)
        print(numCatStories,"G")
    print("G_end")
    return RSS_data


# print(times)
# print(RSS_data.key)
if __name__ == "__main__":
    RSS_data = news_today_dic()
    today=datetime.date.today()
    formatted_today='20' + today.strftime('%y%m%d')
    file_name = "news_in_" + formatted_today + ".json"
    with open(file_name, 'w') as f:
        json.dump(RSS_data,f)
    print("successfully write for " + file_name)
