import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def retrieve_lyrics(artist, song):
    '''
    Takes an Artist name and Song title to retrieve lyrics from www.lyrics.com

    Parameters
    ---------------
    artist : String - artist name
    song : String - song name

    returns array of lyrics
    '''
    base_artist = 'http://www.lyrics.com/artist/'
    base = 'http://www.lyrics.com/'

    if len(artist.split()) < 2:
        url = base_artist + artist.lower()
    else:
        url = base_artist + artist.split(' ')[0].lower()
        for i in range(1, len(artist.split())):
            url = url + '%20' + artist.split(' ')[i].lower()
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    songs = []

    #albums = soup.find_all('h3', {'class': 'artist-album-label'})
    #for alb in albums:
    #    albums.append(alb.text)

    sngs = soup.find_all('td', {'class': 'tal qx'})
    for sng in sngs[::2]:
        if sng.text.lower() == song.lower():
            lyric_url = base + sng.a.attrs['href']
            r = requests.get(lyric_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            lyrics = soup.find('pre', {'id': 'lyric-body-text'}).text
            return np.array(lyrics)
    return 'No Lyrics Found'

print(retrieve_lyrics('Declan Mckenna', 'Rapture'))




    
