
#https://towardsdatascience.com/collecting-news-articles-through-rss-atom-feeds-using-python-7d9a65b06f70
#https://towardsdatascience.com/scrape-and-summarize-news-articles-in-5-lines-of-python-code-175f0e5c7dfc

from newspaper import Article
article = Article('https://www.bbc.co.uk/news/uk-politics-51104386')
article.download()
article.parse()
article.nlp()
