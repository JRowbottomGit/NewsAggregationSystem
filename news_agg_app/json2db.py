from app import db
from app.models import User, Post,News
import json
from string_pro import str_pro

def dbimport(json_path):
    # with open('/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json','r') as f:
    with open(json_path,'r') as f:
        data = json.load(f)

    titles = []
    news = News.query.all()
    for new in news:
        titles.append(new.title)
    outlets = data.keys()
    for outlet in outlets:
        if outlet == 'BBC':
            categorys = data[outlet].keys()
            for category in categorys:
                links = data[outlet][category]
                for link in links:
                    title = data[outlet][category][link]['title']
                    summary = data[outlet][category][link]['summary']
                    # print(News.query.filter(News.title.startswith(title)))
                    if title not in titles:
                        titles.append(title)
                        n = News(outlet = outlet,category = category,title = title,link= link,summary = summary)
                        db.session.add(n)
                        # print(n)
                        db.session.commit()
        elif outlet =='theguardian':
            categorys = data[outlet].keys()
            for category in categorys:
                links = data[outlet][category]
                for link in links:
                    title = data[outlet][category][link]['title']
                    summary = data[outlet][category][link]['summary']
                    summary = str_pro(summary)
                    # print(News.query.filter(News.title.startswith(title)))
                    if title not in titles:
                        titles.append(title)
                        n = News(outlet = outlet,category = category,title = title,link= link,summary = summary)
                        db.session.add(n)
                        # print(n)
                        db.session.commit()

if __name__ == "__main__":
   # json_path =  '/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json'
   # dbimport(json_path)
# print(titles)
# outlet =
# category =
# title = primary_key=True
# link =
# summary =
