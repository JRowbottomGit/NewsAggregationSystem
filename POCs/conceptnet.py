import requests
relations=['RelatedTo','IsA','PartOf','AtLocation', 'Causes','HasContext']
words=['tea']
for i in range(1):
    for j in words:
        j=j.replace(' ','_')
        for r in relations[1:2]:
            str='http://api.conceptnet.io//query?start=/c/en/{}&rel=/r/{}'.format(j,r)


            obj=requests.get(str).json()
            lst=[]



            for x in obj['edges']:
                lst.append(((x['end']['label'])))
                words=list(set(words+lst))
#print(words)

words2=['coffee']
relations=['RelatedTo','IsA','PartOf','AtLocation', 'Causes','HasContext']

for i in range(1):
    for j in words2:
        j=j.replace(' ','_')
        for r in relations[1:2]:
            str='http://api.conceptnet.io//query?start=/c/en/{}&rel=/r/{}'.format(j,r)


            obj=requests.get(str).json()
            lst=[]



            for x in obj['edges']:
                lst.append(((x['end']['label'])))
                words2=list(set(words2+lst))
#print(words2)
ranking=[]
for i in words[0:4]:
    for j in words2[0:4]:
        i=i.replace(' ','_')
        j=j.replace(' ','_')
        #print(i)
        #print(j)
        str='http://api.conceptnet.io/relatedness?node1=/c/en/{}&node2=/c/en/{}'.format(i,j)
        obj=requests.get(str).json()
        v=float(obj['value'])
        ranking.append([v,i,j])
        ranking.sort(reverse=True)
print(ranking[0][1])









