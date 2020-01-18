#https://bigl.es/friday-fun-get-the-news-with-rss-and-gui-zero/
#https://sourceforge.net/projects/xming/ for windows 10 ubuntu gui

import feedparser
from guizero import App, Text, Picture, PushButton
import json
import URL_RSS

RSS_data = {'BBC':{}}

for key,value in URL_RSS.URL_RSS_dict['BBC'].items():
    news_rss = feedparser.parse(value[1])

    RSS_data['BBC'][key] = {}
    for i in range(10):
        id = news_rss['entries'][i]['id']
        RSS_data['BBC'][key][id] = {}
        RSS_data['BBC'][key][id]['link'] = news_rss['entries'][i]['link']
        RSS_data['BBC'][key][id]['title'] = news_rss['entries'][i]['title']
        RSS_data['BBC'][key][id]['published'] = news_rss['entries'][i][''published'']
        RSS_data['BBC'][key][id]['summary'] = news_rss['entries'][i]['summary']
        RSS_data['BBC'][key][id]['story'] =


with open('RSS_data.json', 'w') as f:
    json.dump(RSS_data,f)
