from news_import import news_today_dic
import datetime
import json

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
