# importing the necessary packages
import requests
from bs4 import BeautifulSoup


url = "https://www.bbc.co.uk/news/world-middle-east-51104579"


r1 = requests.get(url)
coverpage = r1.content

soup1 = BeautifulSoup(coverpage, 'lxml')#'html.parser') #''html5lib')
#print(soup1.prettify())

coverpage_news = soup1.find_all('p', class_='story-body__introduction')
print(coverpage_news[0].get_text())

coverpage_news1 = soup1.find_all('h2', class_='story-body__crosshead')
for i in range(len(coverpage_news1)):
    print(coverpage_news1[i].get_text())


#class="story-body__h1"
#class="story-body__introduction"
#class="story-body__crosshead"
#story-body
