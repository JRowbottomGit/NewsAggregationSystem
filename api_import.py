
import pprint
import requests     # 2.19.1
import json #for saving file

secret = 'a24e63a8df8b4c4092756ca4e694c239'

# Define the endpoint
url = 'https://newsapi.org/v2/everything?'

# Specify the query and number of returns
parameters = {
    'q': 'War', #'Gold', #'natural language processing', #'Trump', #'big data', # query phrase
    'pageSize': 20,  # maximum is 100
    'apiKey': secret # your own API key
}

# Make the request
response = requests.get(url, params=parameters)

# Convert the response to JSON format and pretty print it
response_json = response.json()
pprint.pprint(response_json)

#save data json
with open("response_json.json", "w") as f:
    json.dump(response_json, f)

for i in response_json['articles']:
    print(i['title'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create an empty string
text_combined = ''
# Loop through all the headlines and add them to 'text_combined'
for i in response_json['articles']:
    text_combined += i['title'] + ' ' # add a space after every headline, so the first and last words are not glued together
# Print the first 300 characters to screen for inspection
print(text_combined[0:300])

wordcloud = WordCloud(max_font_size=40).generate(text_combined)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
