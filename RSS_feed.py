
#https://bigl.es/friday-fun-get-the-news-with-rss-and-gui-zero/

import feedparser
from guizero import App, Text, Picture, PushButton

BBCnews = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml?edition=uk")
SKYnews = feedparser.parse("http://feeds.skynews.com/feeds/rss/uk.xml")

app = App(title="News Roundup", width=700, height=450)

#BBC_Logo = Picture(app, image="bbcnews.gif")
for i in range(3):
    Text(app, text = BBCnews["entries"][i]["title"], size=16, font="Arial", color="black")

#SKY_Logo = Picture(app, image="sky-news-logo.gif")
for i in range(3):
    Text(app, text = SKYnews["entries"][i]["title"], size=16, font="Arial", color="black")

Close = PushButton(app, command=app.destroy, text="Close News")

app.display()

###################

NewsFeed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml?edition=uk")

entry = NewsFeed.entries[1]

print(entry.published)
print("******")
print(entry.summary)
print("------News Link--------")
print(entry.link)
