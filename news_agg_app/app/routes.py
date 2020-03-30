from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import LoginForm
from flask_login import current_user, login_user
from app.models import User
from flask_login import logout_user
from flask_login import login_required
from flask import request
from werkzeug.urls import url_parse
from app import db
from app.forms import RegistrationForm
from datetime import datetime
from app.forms import EditProfileForm
from app.forms import PostForm
from app.models import Post, News_agg, News
from app import json2db

from PIL import Image
import urllib
import urllib.request

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(body=form.post.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post is now live!')
        return redirect(url_for('index'))
    page = request.args.get('page', 1, type=int)
    posts = current_user.followed_posts().paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('index', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('index', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('index.html', title='Home', form=form,
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)

@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('user', username=user.username, page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('user', username=user.username, page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('user.html', user=user, posts=posts.items,
                           next_url=next_url, prev_url=prev_url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile',
                           form=form)

@app.route('/follow/<username>')
@login_required
def follow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot follow yourself!')
        return redirect(url_for('user', username=username))
    current_user.follow(user)
    db.session.commit()
    flash('You are following {}!'.format(username))
    return redirect(url_for('user', username=username))

@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot unfollow yourself!')
        return redirect(url_for('user', username=username))
    current_user.unfollow(user)
    db.session.commit()
    flash('You are not following {}.'.format(username))
    return redirect(url_for('user', username=username))

@app.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('explore', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('explore', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template("index.html", title='Explore', posts=posts.items,
                          next_url=next_url, prev_url=prev_url)

@app.route('/news_sites')
@login_required
def news_sites():
    news_agg = News_agg()
    news = news_agg.get_news()
    categories = []
    # categories = dict()
    stories = []
    nice_dict = dict()

    ### Below commented out is for the dictionary with a date at the top ###

    # for date in news.keys():
    #     for outlet in news[date]:
    #         print(f'News outlet: {outlet}')
    #         for category in news[date][outlet]:
    #             categories.append(category)
    #             for story in news[date][outlet][category]:
    #                 link = news[date][outlet][category][story]['link']
    #                 title = news[date][outlet][category][story]['title']
    #                 summary = news[date][outlet][category][story]['summary']
    #                 stories.append((category,link,title,summary))
    #             new_list = []
    #             for item in stories:
    #                 if item[0] == category:
    #                     coupled = (item[1],item[2],item[3])
    #                     new_list.append(coupled)
    #             nice_dict[category[0].upper() +category[1:]] = new_list

    for outlet in news:
        print(f'News outlet: {outlet}')
        for category in news[outlet]:
            categories.append(category)
            for story in news[outlet][category]:
                link = news[outlet][category][story]['link']
                title = news[outlet][category][story]['title']
                summary = news[outlet][category][story]['summary']
                stories.append((category,link,title,summary))
            new_list = []
            for item in stories:
                if item[0] == category:
                    coupled = (item[1],item[2],item[3])
                    new_list.append(coupled)
            nice_dict[category[0].upper() +category[1:]] = new_list

    return render_template("news_sites.html", title='News', story_news = nice_dict, categories_new = categories)

@app.route('/category/<category>')
@login_required
def category(category):
    ''' without using database create view 1'''
    aaa = category
    print(f'category is {aaa}')
    news_agg = News_agg()
    news = news_agg.get_news()

    stories = []
    nice_dict = dict()

    ### this part is for the BBC scrape because the categories have lower case letters in the json file ###

    #### This part for the Guardian etc. because categories are capitalised in the json file ####

    ### Below is for the dictionary with a date ####

    # category = category[0].upper() + category[1:]
    # for date in news.keys():
    #     outlet_relevant = list(news[date].keys())
    #     for outlet in news[date]:
    #         for category2 in news[date][outlet]:
    #             for story in news[date][outlet][category2]:
    #                 link = news[date][outlet][category2][story]['link']
    #                 title = news[date][outlet][category2][story]['title']
    #                 summary = news[date][outlet][category2][story]['summary']
    #                 summary = ''.join(summary.split('<p>')).split('</p>')
    #                 summary = ''.join(summary)
    #                 if len(summary) > 200:
    #                     summary = summary[0:200] + '...'
    #                 # Put the outlet in... watch out below may be wrong
    #                 stories.append((category2,link,title,summary, outlet_relevant[1]))
    #             new_list_capitalised = []
    #             for item in stories:
    #                 if item[0] == category: # or category[0].upper() + category[1:]
    #                     coupled = (item[1],item[2],item[3], item[4])
    #                     new_list_capitalised.append(coupled)
    #             new_list_capitalised.reverse()
    #             new_list_capitalised = new_list_capitalised[0:100]

    category = category[0].upper() + category[1:]

    outlet_relevant = list(news.keys())
    for outlet in news:
        for category2 in news[outlet]:
            for story in news[outlet][category2]:
                link = news[outlet][category2][story]['link']
                title = news[outlet][category2][story]['title']
                # pic_url = news[outlet][category2][story]['pic']
                # if pic_url is not None:
                #     pic_filename, _ = urllib.request.urlretrieve(pic_url)
                #     pic = Image.open(pic_filename)
                summary = news[outlet][category2][story]['summary']
                summary = ''.join(summary.split('<p>')).split('</p>')
                summary = ''.join(summary)
                if len(summary) > 200:
                    summary = summary[0:200] + '...'
                # Put the outlet in... watch out below may be wrong
                stories.append((category2,link,title,summary, outlet_relevant[1]))
    new_list_capitalised = []
    for item in stories:
        if item[0] == category: # or category[0].upper() + category[1:]
            coupled = (item[1],item[2],item[3], item[4])
            new_list_capitalised.append(coupled)
    new_list_capitalised.reverse()
    new_list_capitalised = new_list_capitalised[0:100]

    ## Below is for when there is a date in the dictionary ####

    # category = category.lower()
    # for date in news.keys():
    #     outlet_relevant = list(news[date].keys())
    #     for outlet in news[date]:
    #         for category2 in news[date][outlet]:
    #             for story in news[date][outlet][category2]:
    #                 link = news[date][outlet][category2][story]['link']
    #                 title = news[date][outlet][category2][story]['title']
    #                 summary = news[date][outlet][category2][story]['summary']
    #                 summary = ''.join(summary.split('<p>')).split('</p>')
    #                 summary = ''.join(summary)
    #                 if len(summary) > 200:
    #                     summary = summary[0:200] + '...'
    #                 stories.append((category2,link,title,summary, outlet_relevant[0]))
    #             new_list_lowercase = []
    #             for item in stories:
    #                 if item[0] == category: # or category[0].upper() + category[1:]
    #                     coupled = (item[1],item[2],item[3], item[4])
    #                     new_list_lowercase.append(coupled)
    #             new_list_lowercase.reverse()
    #             new_list_lowercase = new_list_lowercase[0:100]
    # new_list =  new_list_lowercase + new_list_capitalised

    category = category.lower()

    outlet_relevant = list(news.keys())
    for outlet in news:
        for category2 in news[outlet]:
            for story in news[outlet][category2]:
                link = news[outlet][category2][story]['link']
                title = news[outlet][category2][story]['title']
                # pic_url = news[outlet][category2][story]['pic']
                # if pic_url is not None:
                #     pic_filename, _ = urllib.request.urlretrieve(pic_url)
                #     pic = Image.open(pic_filename)
                summary = news[outlet][category2][story]['summary']
                summary = ''.join(summary.split('<p>')).split('</p>')
                summary = ''.join(summary)
                if len(summary) > 200:
                    summary = summary[0:200] + '...'
            stories.append((category2,link,title,summary, outlet_relevant[0]))
    new_list_lowercase = []
    for item in stories:
        if item[0] == category: # or category[0].upper() + category[1:]
            coupled = (item[1],item[2],item[3], item[4])
            new_list_lowercase.append(coupled)
    new_list_lowercase.reverse()
    new_list_lowercase = new_list_lowercase[0:100]
    new_list =  new_list_lowercase + new_list_capitalised

    return render_template("category.html", category = aaa, news = new_list)

@app.route('/category2/<category>')
@login_required
def category2(category):
    ''' create view 1 using database '''
    aaa = category
    json_path =  '/Users/alfredtingey/news-aggregation-system-Iteration3/news_archive/Backupnews_in_20200326.json'
    json2db.dbimport(json_path)
    #page = request.args.get('page', 1, type=int)
    allnew = News.query.filter(News.category.startswith(category[0])).all()  #.paginate(page, 100, False)

    ## If we want to add pages in uncomment below. Need to figure out pagination + post processing ##

    # if allnew.has_next:
    #     next_url = url_for('category2', category = category, page=allnew.next_num)
    # else:
    #     None
    #
    # if allnew.has_prev:
    #     prev_url = url_for('category2', category = category, page=allnew.prev_num)
    # else:
    #     None

    new_list = []
    # print("now is test")
    for new in list(allnew):
        link = new.link
        title = new.title
        summary = new.summary
        if len(summary) > 300:
            summary = summary[0:300] + '...'
        ## to deal with cartoons ##
        elif 'cartoon' in title:
            title = title.replace('- cartoon','')
            summary = f'Cartoon: {title}'
        outlet = new.outlet
        new_list.append((link,title,summary,outlet))
    new_list.reverse()

    # print(new_list)
    return render_template("category2.html", category = aaa, news = new_list[0:100])
