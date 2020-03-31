# from news_import import news_today_dic
import datetime
import json
import schedule
import time
import feedparser
# import URL_RSS
# import GetFullText
# import time
import datetime
import news_store
from json2db import dbimport
URL_RSS_dict = {

'BBC':
{
'politics':('https://www.bbc.co.uk/news/politics','http://feeds.bbci.co.uk/news/politics/rss.xml'),
'business':('https://www.bbc.co.uk/news/business','http://feeds.bbci.co.uk/news/business/rss.xml'),
'science':('https://www.bbc.co.uk/news/science_and_environment','http://feeds.bbci.co.uk/news/science_and_environment/rss.xml'),
'technology':('https://www.bbc.co.uk/news/technology','http://feeds.bbci.co.uk/news/technology/rss.xml'),
'sport':('https://www.bbc.co.uk/sport','http://feeds.bbci.co.uk/sport/rss.xml'),
'entertainment':('https://www.bbc.co.uk/news/entertainment_and_arts','http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml')
},

'theguardian':
{
'politics' : ("https://www.theguardian.com/politics","https://www.theguardian.com/politics/rss"),
'business' : ("https://www.theguardian.com/uk/business","https://www.theguardian.com/uk/business/rss"),
'science' : ("https://www.theguardian.com/science","https://www.theguardian.com/science/rss"),
'tech' : ("https://www.theguardian.com/uk/technology","https://www.theguardian.com/uk/technology/rss"),
'sport' : ("https://www.theguardian.com/uk/sport","https://www.theguardian.com/uk/sport/rss"),
'entertaniment' : ("https://www.theguardian.com/uk/culture","https://www.theguardian.com/uk/culture/rss")
},

'telegraph':
{
'politics' : ("https://www.telegraph.co.uk/politics/","https://www.telegraph.co.uk/politics/rss.xml"),
'business' : ("https://www.telegraph.co.uk/business/","https://www.telegraph.co.uk/business/rss.xml"),
'science' : ("https://www.telegraph.co.uk/science/","https://www.telegraph.co.uk/science/rss.xml"),
'tech' : ("https://www.telegraph.co.uk/technology/","https://www.telegraph.co.uk/technology/rss.xml"),
'sport' : ("https://www.telegraph.co.uk/sport/","https://www.telegraph.co.uk/sport/rss.xml"),
'entertaniment' : ("https://www.telegraph.co.uk/culture/","https://www.telegraph.co.uk/culture/rss.xml")
}
}

import requests
from lxml import html

def GetFullTextForBBC(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//div[@class = "story-body__inner"]//p/text()')
    result1 = tree.xpath('//div[@class = "story-body__inner"]//p//a/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
    if sttrr == "":
        result = tree.xpath('//div[@id = "story-body"]//p/text()')
        result1 = tree.xpath('//div[@id = "story-body"]//p//a/text()')
        sttrr = ''
        j = 0
        for ele in range(len(result)):
            if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
                sttrr = sttrr + result[ele] +"\n"
            else:
                # print(result[ele])
                sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
                j = j+1
    return sttrr

def GetFullTextForBBCSimple(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//div[@class = "story-body__inner"]//p/text()')
    result1 = tree.xpath('//div[@class = "story-body__inner"]//p//a/text()')
    sttrr = ''
    result = result + result1
    for ele in range(len(result)):
        sttrr = sttrr + result[ele] +"\n"
    return sttrr


def GetFullTextForGuardian(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//*[@id="article"]/div[2]/div/div[1]/div/p/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        sttrr = sttrr + result[ele] +"\n"
    return sttrr

def GetFullTextForABC(url:str) ->str:
    # print(1)
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)

    result = tree.xpath('//article[@class="Article__Content story"]/p/text()')
    result1 = tree.xpath('//article[@class="Article__Content story"]/p//a/text()')
    # print(result)
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        # if
        if len(result[ele])<2:
            pass
        elif result[ele][1].isupper() or result[ele][1] == '"' or result[ele][1] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            # print(result[ele])
            # print('??'+result[ele][0]+'??')
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
            # print(j)
    return sttrr


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

    for key,value in URL_RSS_dict['BBC'].items():
        news_rss = feedparser.parse(value[1])
        RSS_data['BBC'][key] = {}
        numCatStories = len(news_rss['entries'])
        # start = time.time()
        for i in range(numCatStories):                         #####################################change this to the length of each topic
            try:
                url = news_rss['entries'][i]['link']
                sttrr = GetFullTextForBBC(url)
            except:
                sttrr = GetFullTextForBBCSimple(url)
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

    for key,value in URL_RSS_dict['theguardian'].items():
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
            RSS_data['theguardian'][key][id]['story'] = GetFullTextForGuardian(url)
            # print(i,numCatStories)
        # test_time = end - start
        # end = time.time()
        # times["theguardian"+key] = (numCatStories,test_time)
        print(numCatStories,"G")
    print("G_end")
    return RSS_data

def news_store():
    today=datetime.date.today()
    # print(today)
    formatted_today='20' + today.strftime('%y%m%d')
    # print(formatted_today)
    today_news = news_today_dic()

    back_up_file_name = '/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/Backup' + "news_in_" + formatted_today + ".json"
    with open(back_up_file_name, 'w') as f:
        json.dump(today_news,f)
    print("successfully write for " + back_up_file_name)

    with open('/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/news_history.json', 'r') as f:
        history_data = json.load(f)

    history_data[formatted_today] = today_news

    with open('/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/news_history.json', 'w') as f:
        json.dump(history_data,f)

    print("finish updating")
    return back_up_file_name


def schedule_import():
    json_path = news_store()
    dbimport(json_path)
schedule.every().day.at("15:56").do(schedule_import)



if __name__ == "__main__":
    while True:
        # Checks whether a scheduled task
        # is pending to run or not
        schedule.run_pending()
        time.sleep(61)
    # json_path = news_store()
    # dbimport(json_path)
