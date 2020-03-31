import json
def str_pro(s0):
    sout = ''
    activate = False
    for char in s0:
        if char == '<':
            activate = True
        if char == '>':
            activate = False
            # break
        if (not activate) and (char !='>'):
            sout = sout + char
    return sout.replace('Continue reading...','')
    
if __name__ == '__main__':
    with open('/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json','r') as f:
        data = json.load(f)


    print(data['theguardian'].keys())

    # print(data['theguardian']['Entertaniment & Arts']['https://www.theguardian.com/film/2020/feb/07/sam-mendes-1917-best-director-oscars-male-dominated'].keys())
    # print(data['theguardian']['Entertaniment & Arts']['https://www.theguardian.com/film/2020/feb/07/sam-mendes-1917-best-director-oscars-male-dominated']['title'])

    sss = data['theguardian']['Entertaniment & Arts']['https://www.theguardian.com/film/gallery/2020/feb/03/baftas-2020-behind-the-scenes-in-pictures']['summary']
    # print(data['theguardian']['Entertaniment & Arts'].keys())
    print('')

    print(sss)
    print('')
    print('')
    print(str_pro(sss))
