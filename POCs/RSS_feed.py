
#https://bigl.es/friday-fun-get-the-news-with-rss-and-gui-zero/
#https://sourceforge.net/projects/xming/ for windows 10 ubuntu gui

import feedparser
from guizero import App, Text, Picture, PushButton
import json
import URL_RSS


BBCnews = feedparser.parse("http://feeds.bbci.co.uk/news/politics/rss.xml")
app = App(title="News Roundup", width=700, height=450)
#BBC_Logo = Picture(app, image="bbcnews.gif")
for i in range(20):
    #Text(app, text = BBCnews["entries"][i]["title"], size=16, font="Arial", color="black")
    #Text(app, text = BBCnews["entries"][i]["summary"], size=16, font="Arial", color="black")
    Text(app, text = BBCnews["entries"][i]["link"], size=16, font="Arial", color="black")


# SKYnews = feedparser.parse("http://feeds.skynews.com/feeds/rss/uk.xml")
# #SKY_Logo = Picture(app, image="sky-news-logo.gif")
# for i in range(3):
#     Text(app, text = SKYnews["entries"][i]["title"], size=16, font="Arial", color="black")

Close = PushButton(app, command=app.destroy, text="Close News")
app.display()
###################

NewsFeed = feedparser.parse("http://feeds.bbci.co.uk/news/politics/rss.xml")
print(NewsFeed.entries)
# entry = NewsFeed.entries[1]
# print(entry)
# print(entry.published)
# print("******")
# print(entry.summary)
# print("------News Link--------")
# print(entry.link)
