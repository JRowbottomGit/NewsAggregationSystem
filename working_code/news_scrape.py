# importing the necessary packages
#https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558
# https://www.guru99.com/selenium-python.html

import requests
from bs4 import BeautifulSoup


url = "https://www.bbc.co.uk/news/world-middle-east-51104579"


r1 = requests.get(url)
coverpage = r1.content

soup1 = BeautifulSoup(coverpage, 'lxml')#'html.parser') #''html5lib')
#print(soup1.prettify())


# coverpage_news = soup1.find_all('p', class_='story-body__introduction')
# for i in range(len(coverpage_news)):
#     print(coverpage_news[i].get_text())
# # print(coverpage_news[0].get_text())
#
#
# coverpage_news1 = soup1.find_all('p', class_='story-body__inner')
# for i in range(len(coverpage_news1)):
#     print(coverpage_news1[i].get_text())


coverpage_news3 = soup1.find_all('div', class_='story-body').p.text

#print(coverpage_news3)

paragraph = coverpage_news3.p.text

for i in range(len(coverpage_news3)):
    print(coverpage_news3[i].get_text())


# coverpage_news2 = soup1.find_all('h2', class_='story-body__crosshead')
# for i in range(len(coverpage_news2)):
#     print(coverpage_news2[i].get_text())



#class="story-body__h1"
#class="story-body__introduction"
#class="story-body__crosshead"
#story-body
