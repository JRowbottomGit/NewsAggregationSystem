import json
def add_to_archive():
    json_list = [
    'Backupnews_in_20200326',
    'Backupnews_in_20200327',
    'Backupnews_in_20200329',
    'Backupnews_in_20200330']
    #with open('/vol/project/2019/545/g1954505/news-aggregation-system/news_archive/news_history.json', 'r') as f:
    with open('../news_archive/news_history.json', 'r') as f:
        history_data = json.load(f)

    for day in json_list:
        with open('../news_archive/' + day + '.json', 'r') as f:
            today_news = json.load(f)
        print(day)
        print(day[-8:])
        history_data[day[-8:]] = today_news
    #
    with open('../news_archive/news_history.json', 'w') as f:
        json.dump(history_data,f)
    print(history_data.keys())


#scp /mnt/c/Users/jabro/Documents/MSc\ Imperial\ -\ AI/GroupProject/news-aggregation-system/news_archive/news_history.json jr1419@shell4.doc.ic.ac.uk://vol/project/2019/545/g1954505/news-aggregation-system/news_archive

add_to_archive()
# if __name__ = "__main__"
