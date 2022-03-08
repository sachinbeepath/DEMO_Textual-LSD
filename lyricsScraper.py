import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import time

clear = lambda: os.system('cls')

def printc(x):
    clear()
    print(x)
    return

class LyricScraper():
    def __init__(self, data_frame, artist_col, song_col, ind=None, no_lyrics_return = 'No Lyrics Found', save=False, file_name=None, add_to_df=False):
        '''
        Takes a pandas dataframe with artist and song title info and retrieves lyrics from www.lyrics.com

        Parameters
        ----------------
        data_frame : DataFrame - pandas dataframe with song info
        artist_col : String - title of column with artist name
        song_col : String - title of column with song name
        ind : Array<int> - array of indices to look at when searching from lyrics
        no_lyrics_return : Any - what to return if no lyrics are found
        save : Bool - whether to save the lyrics/artist/song into a new csv or not
        file_name : String - if saving, what should the file be called? (Inlude extension)
        add_to_df : Bool - whether to add a lyrics column to the original dataframe
        '''
        self.df = data_frame
        self.no_lyr = no_lyrics_return
        self.save = save
        self.file_name = file_name
        self.artist_col = artist_col
        self.song_col = song_col
        self.add_df = add_to_df
        self.symbol_lookup = {"'": "%27", '/': '-', '&': '%26', ',': '%2C', '.': '.-'}
        self._lookup_times = []
        self._ETC = 0

        artists = self.df[self.artist_col]
        songs = self.df[self.song_col]
        
        assert len(artists) == len(songs), 'Number of artists does not match number of songs'

        self.ind = np.arange(0, len(songs), 1) if ind == None else ind
        artists = artists[self.ind]
        songs = songs[self.ind]

        self.lyrics = []
        for i, (artist, song) in enumerate(zip(artists, songs)):
            t = time.time()
            if len(self._lookup_times) > 20:
                av_t = np.average(self._lookup_times[-20:])
                self._ETC = (len(artists) - i) * av_t

            printc(f'Scraping Lyrics ({i}/{len(artists)}) --- Estimated Time Remaining: {self._ETC:.0f}seconds')
            self.lyrics.append(self.retrieve_lyrics(artist, song, self.no_lyr))
            self._lookup_times.append(time.time() - t)

        self.lyrics = np.array(self.lyrics)

        if self.save:
            assert self.file_name != None, "Invalid file name for saving"
            self.save_df(self.df, self.file_name)
        
        if self.add_df:
            assert ind == None, 'Cannot add subsection of data to dataframe - ind must be None'
            self.add_to_df()
        
    def get_lyrics(self):
        return self.lyrics
        
    def retrieve_lyrics(self, artist, song, no_lyric_return):
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

        artist = str(artist) #cus some idiots name their band a number...
        song = str(song) #and some songs too i guess...
        if len(artist.split()) < 2:
            url = base_artist + self.convert_symbols(artist)
        else:
            url = base_artist + self.convert_symbols(artist).split(' ')[0]
            for i in range(1, len(artist.split())):
                url = url + '%20' + self.convert_symbols(artist).split(' ')[i]
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        songs = []

        #albums = soup.find_all('h3', {'class': 'artist-album-label'})
        #for alb in albums:
        #    albums.append(alb.text)

        sngs = soup.find_all('td', {'class': 'tal qx'})
        if sngs == []:
            # Artist url failed, so try to find artist in search results.
            arts = soup.find_all('td', {'class': 'tal fx'})
            for art in arts:
                if artist.lower() == art.text.lower():
                    url = base + art.a.attrs['href']
                    r = requests.get(url)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    sngs = soup.find_all('td', {'class': 'tal qx'})
                    break
        if sngs == []:    
            print("Artist not found...")
            return no_lyric_return

        for sng in sngs[::2]:
            if sng.text.lower() == song.lower():
                lyric_url = base + sng.a.attrs['href']
                r = requests.get(lyric_url)
                soup = BeautifulSoup(r.text, 'html.parser')
                lyrics = soup.find('pre', {'id': 'lyric-body-text'})
                if lyrics != None:
                    return lyrics.text
                else:
                    print('No Lyrics Available')
                    break
            elif song.lower() in sng.text.lower():
                print('Partial Match Found')
                lyric_url = base + sng.a.attrs['href']
                r = requests.get(lyric_url)
                soup = BeautifulSoup(r.text, 'html.parser')
                lyrics = soup.find('pre', {'id': 'lyric-body-text'})
                if lyrics != None:
                    return lyrics.text
                else:
                    print('No Lyrics Available')
                    break
        return no_lyric_return

    def get_df(self):
        '''Returns DataFrame'''

        return self.df

    def add_to_df(self):
        '''Adds Lyric column to DataFrame'''

        self.df['Lyrics'] = self.lyrics
        return

    def save_df(self, df, file_name):
        '''Creates new DataFrame(Artist, Song, Lyrics) and saves csv'''

        assert isinstance(df, pd.DataFrame), 'Is not a pandas DataFrame'
        new_df = df
        df['Lyrics'] = self.lyrics
        df.to_csv(file_name)
        return

    def convert_symbols(self, text):
        '''Replaces special characters with their URL counterpart'''
        #for key in self.symbol_lookup.keys():
        #    text = text.replace(key, self.symbol_lookup[key])

        return text



#print(retrieve_lyrics('Declan Mckenna', 'Rapture'))




    
