from django.shortcuts import render
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from django.conf import settings
import json
from django.http import JsonResponse
from django.contrib.staticfiles import finders

CLIENT_ID = ""
CLIENT_SECRET = ""

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                           client_secret=CLIENT_SECRET))

def dataframex():
    csv_path = finders.find('data/data.csv')
    data_csv = pd.read_csv(csv_path)
    print(data_csv)

genre_data = os.path.join(os.path.dirname(__file__), 'data_by_genres.csv')
year_data = os.path.join(os.path.dirname(__file__), 'data_by_year.csv')
artist_data = os.path.join(os.path.dirname(__file__), 'data_by_artist.csv')

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        print('Fetching song information from local dataset')
        return song_data

    except IndexError:
        print('Fetching song information from spotify dataset')
        return find_song(song['name'], song['year'])
    
def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))#nd-array where n is number of songs in list. It contains all numerical vals of songs in sep list.
    #print(f'song_matrix {song_matrix}')
    return np.mean(song_matrix, axis=0) # mean of each ele in list, returns 1-d array

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = [] # 'name', 'year'
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value) # creating list of values
    return flattened_dict

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=25,
                                   verbose=False))
                                 ], verbose=False)

def recommend_songs(song_list, spotify_data, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    #print(f'song_center {song_center}')
    
    # Fit the scaler on the training data
    scaler = song_cluster_pipeline.steps[0][1]  # Assuming song_cluster_pipeline is a pipeline containing the scaler
    scaler.fit(spotify_data[number_cols])

    # Transform the data using the fitted scaler
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    #print(f'distances {distances}')
    
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def main(request):
    if request.method == 'POST':
        user_input = request.POST.get('music_input')
        # Fetch actual Spotify data (modify this based on your actual Spotipy usage)

        data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'))

        user_input_list = [item.strip() for item in user_input.split('/')]
        
        if len(user_input_list) == 2:
            user_input_dict_list = [{'name': item.split(',')[0].strip(), 'year': int(item.split(',')[1].strip())} for item in user_input_list]
            recommended_songs = recommend_songs(user_input_dict_list, data)
            data = {
                "recommended_songs": recommended_songs,
                "user_input": user_input,
            }
            return render(request, "main.html", data)
        
    return render(request, "main.html")

def about(request):
    return render(request, "about.html")
