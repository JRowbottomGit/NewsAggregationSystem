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
        if result[ele][0].isupper() or result[ele][0] == '"':
            sttrr = sttrr + result[ele] +"\n"
        else:
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


if __name__ == "__main__":
    # url = "https://www.bbc.co.uk/news/world-middle-east-51104579"
    # url = "https://www.bbc.co.uk/news/world-asia-51166339"
    # print(GetFullTextForBBC(url))
    # url = "https://www.theguardian.com/world/2020/jan/19/snowmageddon-cleanup-begins-after-record-newfoundland-storm"
    url ="https://www.theguardian.com/sport/2020/jan/18/saracens-relegated-end-of-season-premiership-rugby"
    print(GetFullTextForGuardian(url))
