import profile
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import sqlite3
import requests
import re
from collections import defaultdict
from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from surprise import dump
import pickle
import random
import numpy as np
import pandas as pd
from Evaluator import Evaluator
from MovieLensUser import MovieLens
import os
import os.path
import csv
import shutil

conn = sqlite3.connect('data.db')  # Creating the connection to database
c = conn.cursor()

# Functions for using the database


def create_usertable():
    c.execute(
        'CREATE TABLE IF NOT EXISTS userstable(name TEXT,username TEXT,password TEXT)')


def create_movietable():
    c.execute('CREATE TABLE IF NOT EXISTS movietable(username TEXT,data BLOB)')


def add_userdata(name, username, password):
    l = list()
    l.append(password)
    hashed_l = stauth.Hasher(l).generate()
    c.execute('INSERT INTO userstable(name,username,password) VALUES (?,?,?)',
              (name, username, hashed_l[0]))
    conn.commit()


def add_moviedata(username, data):
    c.execute('INSERT INTO movietable(username,data) VALUES (?,?)',
              (username, data))
    conn.commit()


def check_entry(username):
    c.execute('SELECT * FROM movietable WHERE username=?', (username,))
    rows = c.fetchall()
    if len(rows) == 1:
        return True
    else:
        return False


def get_moviedata(username):
    c.execute('SELECT data FROM movietable WHERE username=?', (username,))
    rows = c.fetchall()
    data = pickle.loads(rows[0][0])
    return data

# Functions for getting movie data from TMDB API
def update_moviedata(username, data):
    c.execute('UPDATE movietable SET data=? WHERE username=?', (data, username))
    conn.commit()


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=f3439b10af2bac0609b650406b6e3a7d&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def fetch_name(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=f3439b10af2bac0609b650406b6e3a7d&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    name = data['title']
    return name


def LoadMovieLensData():
    ml = MovieLens()
    # print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    # print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = defaultdict(int)
    return (ml, data, rankings)


def copy_csv():
    shutil.copy2('ml-latest-small/ratings.csv',
                 'ml-latest-small/ratingsAfterUser.csv')


with st.sidebar:
    choose = option_menu("Menu", ["Home", "Recommendations", "Profile", "Signup"],
                         icons=['house', 'fullscreen',
                                'person', 'box-arrow-right'],
                         menu_icon="list", default_index=0)


if choose == "Home" or choose == "Profile" or choose == 'Recommendations':
    name_list = list()
    username_list = list()
    pass_list = list()
    for row in c.execute('SELECT * FROM userstable ORDER BY name'):
        name_list.append(row[0])
        username_list.append(row[1])
        pass_list.append(row[2])
    authenticator = stauth.Authenticate(
        name_list,
        username_list,
        pass_list,
        'some_cookie_name',
        'some_signature_key',
        30
    )

    name, authentication_status, username = authenticator.login(
        'Login', 'main')
    movies = pickle.load(open('movie_list.pkl', 'rb'))

    if authentication_status and choose == 'Home':
        authenticator.logout('Logout', 'main')
        st.title('Home')
        st.write(
            f'Welcome, {name}. Rate atleast 5 movies to get movie recommendations')
        with st.form("RatingForm"):
            movie_list = movies['title'].values
            selected_movie = st.selectbox(
                "Type or select the movie you want to rate from the dropdown", movie_list)
            rating = st.slider("Movie Rating", min_value=0.5,
                               max_value=5.0, step=0.5)
            rate = st.form_submit_button("Rate")
            if rate:
                create_movietable()
                if check_entry(username):
                    rating_list = get_moviedata(username)
                    temp_tuple = (611, int(re.findall(
                        r'[0-9]+', str(movies[movies['title'] == selected_movie]))[1]), rating, 0)
                    rating_list.append(temp_tuple)
                    pdata = pickle.dumps(
                        rating_list, protocol=pickle.HIGHEST_PROTOCOL)
                    update_moviedata(username, sqlite3.Binary(pdata))
                else:
                    temp_tuple = (611, int(re.findall(
                        r'[0-9]+', str(movies[movies['title'] == selected_movie]))[1]), rating, 0)
                    rating_list = [temp_tuple]
                    pdata = pickle.dumps(
                        rating_list, protocol=pickle.HIGHEST_PROTOCOL)
                    add_moviedata(username, sqlite3.Binary(pdata))
                st.success("You have rated {}, with {} stars rating".format(
                    selected_movie, rating))

    elif authentication_status and choose == 'Profile':
        authenticator.logout('Logout', 'main')
        st.title('Profile')
        st.subheader('Username: {}'.format(username))
        st.subheader('Rated Movies')
        profile_data = get_moviedata(username)
        profile_df = pd.DataFrame(profile_data, columns=[
                                  'Index', 'Movie Name', 'Rating', 'Drop'])
        profile_df = profile_df.drop(columns=['Index', 'Drop'])
        profile_df['Movie Name'] = profile_df['Movie Name'].apply(
            lambda x: fetch_name(movies[movies['movieId'] == x].values.tolist()[0][3]))
        st.dataframe(profile_df)

    elif authentication_status and choose == 'Recommendations':
        authenticator.logout('Logout', 'main')
        st.title('Recommendations')
        namesNposter = []
        with st.form("RecommendationForm"):
            selected_algo = st.selectbox(
                "Select the algorithm using which you want to receiver the recommendations", ["SVD", "SVD++"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                (pred, algo) = dump.load(selected_algo)

                np.random.seed(0)
                random.seed(0)
                # Load up common data set for the recommender algorithms
                if os.path.isfile('ml-latest-small/ratingsAfterUser.csv'):
                    os.remove('ml-latest-small/ratingsAfterUser.csv')
                copy_csv()
                movie_data = get_moviedata(username)
                with open('ml-latest-small/ratingsAfterUser.csv', 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerows(movie_data)

                (ml, evaluationData, rankings) = LoadMovieLensData()

                # Construct an Evaluator to evaluate them
                evaluator = Evaluator(evaluationData, rankings)

                ml = pickle.load(open('ml.pkl', 'rb'))
                testSet = evaluator.dataset.GetAntiTestSetForUser(611)
                print(len(testSet))
                predictions = algo.test(testSet)
                recommendations = []
                for userID, movieID, actualRating, estimatedRating, _ in predictions:
                    intMovieID = int(movieID)
                    recommendations.append((intMovieID, estimatedRating))

                recommendations.sort(key=lambda x: x[1], reverse=True)
                # print(recommendations)

                for movie in recommendations[:10]:
                    temp_list = movies[movies['movieId']
                                       == movie[0]].values.tolist()
                    tmdbID = int(temp_list[0][3])
                    namesNposter.append(
                        (fetch_name(tmdbID), fetch_poster(tmdbID)))
                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(
                    10)
                with col1:
                    st.image(namesNposter[0][1])
                    st.text(namesNposter[0][0])
                with col2:
                    st.image(namesNposter[1][1])
                    st.text(namesNposter[1][0])
                with col3:
                    st.image(namesNposter[2][1])
                    st.text(namesNposter[2][0])
                with col4:
                    st.image(namesNposter[3][1])
                    st.text(namesNposter[3][0])
                with col5:
                    st.image(namesNposter[4][1])
                    st.text(namesNposter[4][0])
                with col6:
                    st.image(namesNposter[5][1])
                    st.text(namesNposter[5][0])
                with col7:
                    st.image(namesNposter[6][1])
                    st.text(namesNposter[6][0])
                with col8:
                    st.image(namesNposter[7][1])
                    st.text(namesNposter[7][0])
                with col9:
                    st.image(namesNposter[8][1])
                    st.text(namesNposter[8][0])
                with col10:
                    st.image(namesNposter[9][1])
                    st.text(namesNposter[9][0])
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(
            10)

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
elif choose == "Signup":
    st.subheader("Create an Account")
    new_name = st.text_input('First Name')
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type='password')
    if st.button('Signup'):
        if new_name == '' or new_pass == '' or new_user == '':
            st.error(
                'You are not entering one of the fields. All fields are required.')
        else:
            create_usertable()
            add_userdata(new_name, new_user, new_pass)
            st.success(
                "You have successfully created an account. Click on Home to login")
