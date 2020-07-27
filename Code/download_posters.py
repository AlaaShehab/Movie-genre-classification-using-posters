import os.path
import pandas as pd
import urllib.request
import os


def download_poster(movie):
    try:
        response = urllib.request.urlopen(movie[1])
        data = response.read()
        file = open('data/images/'+ movie[0] +'.jpg', 'wb')
        file.write(bytearray(data))
        file.close()
        return data
    except:
        print('error downloading poster')


def download_posters():
    movies = list_movies()
    for movie in movies:
        if (os.path.exists('data/images/'+ movie[0] +'.jpg') == False):
            download_poster(movie)


def list_movies():
    parsed_movies = []
    if len(parsed_movies) == 0:
        url = 'https://raw.githubusercontent.com/AlaaShehab/Movie-genre-classification/master/movies_metadata.csv'
        data = pd.read_csv(url, sep=",", encoding='ISO-8859-1')
        data = data.loc[:, ['genres', 'poster_path', 'imdb_id', 'release_date']]
        data = data.dropna()
        for i, r in data.iterrows():
            movie = parse_row(r)
            if len(movie) == 2:
                parsed_movies.append(movie)
    return parsed_movies


def parse_row(row):
    movie = []
    if len(str(row['release_date'])) <= 4 or len(str(row['imdb_id'])) <= 2 or len(str(row['genres'])) <= 2:
        return movie
    movie.append(str(int(row['imdb_id'][2::])))
    url = str(row['poster_path'])
    if len(url) > 0:
        movie.append('http://image.tmdb.org/t/p/w185/' + url.replace('"', ''))
    return movie


def main():
    original_images_dir = 'data/images/'
    if os.path.isdir(original_images_dir) == False:
        os.makedirs(original_images_dir)
    download_posters()

if __name__ == '__main__':
    main()