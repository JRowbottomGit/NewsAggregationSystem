# Group Project: News Aggregation System - Alfred Tingey, James Rowbottom, Avishkar Mahajan, Zhangrui Chen

The code that we have provided aims to build a news aggregation web application consisting of 3 main views and an additional social media aspect. We have many json data files that get updated through scraping and are in JSON format, which are present in the 'news_agg_app' folder. In the same folder, there are NLP models and a config file. In the POCs folder we have NLP and scripts for scraping news from online outlets. In the 'app' folder we have scripts that build the web application. The main methods in the web application for each view are present in the 'routes.py' script. In the 'models.py' script we create all of the Flask databases. In the 'forms.py' we have the forms that we create, for example the post form, sign up form and comment forms. In the 'templates' folder we have all of the HTML files that the web application consists of. 

Thus the task of the group project was to create a web application that had 3 views in it: 

1) Show a user news articles that are sorted by category
2) Show a user news articles that are sorted by topic
3) Show a user recommended news articles based off of key words that they input into the web application. 

The code provided is split into the follwing sections:

> Importing data in the form of JSON or Pandas
> Creating the databases that the web application uses e.g. News articles, Users, Comments, etc. 
> Methods that do the back-end software engineering and render the templates in the routes.py file.
> HTML templates
> NLP files for topic generation
> NLP files for recommender system 
> Additional features, such as translation, search bar, comments etc.
> Testing files

### Tech

Our code uses the follwing libraries:

* cycler==0.10.0
* kiwisolver==1.1.0
* matplotlib==3.1.1
* numpy==1.17.3
* opencv-python==4.1.1.26
* pyparsing==2.4.4
* python-dateutil==2.8.1
* six==1.13.0
* torch==1.3.0
* wordcloud == 1.6.0
* requests == 2.22.0
* #feedparser / guizero / newspaper3k / beautifulsoup4
* nltk == 3.4.5
* gensim == 3.8.1
* sklearn

* alembic==0.9.6
* Babel==2.5.1
* blinker==1.4
* certifi==2017.7.27.1
* chardet==3.0.4
* click==6.7
* dominate==2.3.1
* elasticsearch==7.5.1
* Flask==1.0.2
* Flask-Babel==0.11.2
* Flask-Bootstrap==3.3.7.1
* Flask-HTTPAuth==3.2.3
* Flask-Login==0.4.0
* Flask-Mail==0.9.1
* Flask-Migrate==2.1.1
* Flask-Moment==0.5.2
* Flask-SQLAlchemy==2.3.2
* Flask-WTF==0.14.2
* guess_language-spirit==0.5.3
* idna==2.6
* itsdangerous==0.24
* Jinja2==2.10
* Mako==1.0.7
* MarkupSafe==1.0
* PyJWT==1.5.3
* python-dateutil==2.6.1
* python-dotenv==0.7.1
* python-editor==1.0.3
* pytz==2017.2
* redis==3.2.1
* requests==2.18.4
* rq==1.0
* six==1.11.0
* SQLAlchemy==1.1.14
* urllib3==1.22
* visitor==0.1.3
* Werkzeug==0.14.1
* WTForms==2.1



### Installation

All of the above libraries need to be installed in order to run our web application. We have included them in a requirements.txt file, therefore to install all of the libraries one has to pip install requirements.txt.

### Instructions for code to be run 

* The first step is to download the required data that will be using, which consists of all of the news articles.
* Next there is the data manipulation using NLP techniques.
* Additionally, one needs to create a virtual environment and download the requirements.txt into the virtual environment.
* Set the Flask environment by typing 'FLASKAPP=news_aggregation.py' into the terminal.
* We use a microsoft translation system in our web application, therefore to run our code you will need to create a Microsoft Azure account and gain a 'translation key'. This can then be set as an environment variable as above. 
* Type flask run into the terminal and it should be good to go. 
