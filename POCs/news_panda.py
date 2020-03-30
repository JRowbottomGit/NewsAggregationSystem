import pandas as pd
import json
import os
#from pandas.io.json import json._normalise

dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)
'''
Creates RSS_data dictionary of the following form:
############this is missing date
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

# user_dict = {12: {'Category 1': {'att_1': 1, 'att_2': 'whatever'},
#                   'Category 2': {'att_1': 23, 'att_2': 'another'}},
#              15: {'Category 1': {'att_1': 10, 'att_2': 'foo'},
#                   'Category 2': {'att_1': 30, 'att_2': 'bar'}}}
# user_dict = {12: {'Category 1': {"story1":{'att_1': 1, 'att_2': 'whatever'}},
#                   'Category 2': {"story1":{'att_1': 23, 'att_2': 'another'}}},
#              15: {'Category 1': {"story1":{'att_1': 10, 'att_2': 'foo'}},
#                   'Category 2': {"story1":{'att_1': 30, 'att_2': 'bar'}}}}

#news_history_path = "/vol/project/2019/545/g1954505/news-aggregation-system/working_code/news_history.json"
#news_history_path = dir_path + "/working_code/news_history.json"
news_history_path = dir_path + "/../working_code/news_history.json"
with open(news_history_path, "r") as f:
    news_history = json.load(f)

#need to add another index for date
#currently its 60 rows x 853 columns

# user_dict = news_history
# df = pd.DataFrame.from_dict({(i,j,k,l): user_dict[i][j][k][l]
#                            for i in user_dict.keys()
#                            for j in user_dict[i].keys()
#                            for k in user_dict[i][j].keys()
#                            for l in user_dict[i][j][k].keys()
#                            },
#                        orient='index')
#print(df)

#another idea with this
#but where the v is need to nest the next item with anoth dict comprehension
#pd.concat({k: pd.DataFrame(v).T for k, v in user_dict.items()}, axis=0)

# table = json_normalise(news_history)

rows = []

for date in news_history.keys():
    for source in news_history[date].keys():
        for category in news_history[date][source].keys():
            for storyid, story_object in news_history[date][source][category].items():
                link = story_object["link"]
                title = story_object["title"]
                published = story_object["published"]
                summary = story_object["summary"]
                row = [date, source, category, storyid, link, title, published, summary]
                rows.append(row)



df = pd.DataFrame.from_records(rows)#, columns)
df = pd.DataFrame(rows, columns = ["Date","Source","Category","story_id","link","title","published","summary"])
print(df)
#print(df.head(5))

table = pd.pivot_table(df, values="story_id", index=["Date","Source"], columns=["Category"], aggfunc="count",margins=True)
print(table)

# pd.DataFrame.from_dict({(i,j): user_dict[i][j]
#                            for i in user_dict.keys()
#                            for j in user_dict[i].keys()},
#                        orient='index')
