import requests
from lxml import html

def GetFullTextForBBC(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//div[@class = "story-body__inner"]//p/text()')
    result1 = tree.xpath('//div[@class = "story-body__inner"]//p//a/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
    if sttrr == "":
        result = tree.xpath('//div[@id = "story-body"]//p/text()')
        result1 = tree.xpath('//div[@id = "story-body"]//p//a/text()')
        sttrr = ''
        j = 0
        for ele in range(len(result)):
            if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
                sttrr = sttrr + result[ele] +"\n"
            else:
                # print(result[ele])
                sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
                j = j+1

    return sttrr



def GetFullTextForGuardian(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//*[@id="article"]/div[2]/div/div[1]/div/p/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        sttrr = sttrr + result[ele] +"\n"
    return sttrr

def GetFullTextForABC(url:str) ->str:
    # print(1)
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)

    result = tree.xpath('//article[@class="Article__Content story"]/p/text()')
    result1 = tree.xpath('//article[@class="Article__Content story"]/p//a/text()')
    # print(result)
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        # if
        if len(result[ele])<2:
            pass
        elif result[ele][1].isupper() or result[ele][1] == '"' or result[ele][1] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            # print(result[ele])
            # print('??'+result[ele][0]+'??')
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
            # print(j)
    return sttrr

if __name__ == "__main__":
    # url = "https://www.bbc.co.uk/news/world-midd`le-east-51104579"
    # url = "https://www.bbc.co.uk/news/world-asia-51166339"
    # url = "https://www.bbc.co.uk/news/uk-politics-51195059"
    # url = "https://www.bbc.co.uk/news/newsbeat-51177738"
    # url = "https://www.bbc.co.uk/sport/tennis/51105556"
    # url = "https://www.bbc.co.uk/sport/athletics/51196920"
    url = "https://www.bbc.co.uk/news/science-environment-51179688"
    print(GetFullTextForBBC(url))
    # GetFullTextForBBC(url)
    # url = "https://www.theguardian.com/world/2020/jan/19/snowmageddon-cleanup-begins-after-record-newfoundland-storm"
    # url ="https://www.theguardian.com/sport/2020/jan/18/saracens-relegated-end-of-season-premiership-rugby"
    # print(GetFullTextForGuardian(url))

    # url = "https://abcnews.go.com/US/major-storm-moves-offshore-cold-air-rushes-western/story?id=68384385&cid=clicksource_4380645_2_heads_hero_live_headlines_hed"
    # url = "https://abcnews.go.com/International/wireStory/afghan-officials-taliban-kill-members-family-68384149?cid=clicksource_4380645_3_mobile_web_only_headlines_headlines_hed"
    # url = "https://abcnews.go.com/US/michigan-man-finds-43000-couch-bought-35/story?id=68369858&cid=clicksource_4380645_3_mobile_web_only_headlines_headlines_hed"
    # url = "https://abcnews.go.com/US/wireStory/lee-working-amend-confederate-general-proclamation-68373024?cid=clicksource_4380645_3_mobile_web_only_headlines_headlines_hed"
    # url = "https://abcnews.go.com/US/top-prosecutor-receives-racist-voicemail-posts-social-media/story?id=68369967&cid=clicksource_4380645_9_heads_posts_headlines_hed"
    # print(GetFullTextForIn(url))
    # print(1)
