import csv
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
import random
import os
import shutil
from shutil import copyfile
import re

def get_labels(expected):
  result = []
  for i in range(20):
    if genres[i] in expected:
      result.append(1)
    else:
      result.append(0)
  return result

def get_movie_less_than_year(data, year):
  list_ = []
  for i, v in data['release_date'].items():
    if str(v) <= 4:
      continue
    v = v[0:4]
    if int(v) <= year:
      list_.append(i)
  return list_

def write_data(movie_train, movie_train_label, movie_validate, movie_validate_label):
  print("writing in file")
  with open('data/train.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(movie_train)
  with open('data/train_label.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(movie_train_label)
  with open('data/validate.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(movie_validate)
  with open('data/validate_label.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(movie_validate_label)

def create_id_label(genres, header):
  original_images_dir = 'data/images/'
  data = pd.read_csv('data/movies_metadata.csv', encoding='ISO-8859-1')
  data = data.loc[:, ['poster_path', 'imdb_id', 'release_date', 'genres']]
  data = data.dropna()
  indices = get_movie_less_than_year(data, 1978)
  data.drop(indices)

  movie_train = []
  movie_train_label = []
  movie_validate = []
  movie_validate_label = []

  movie_train.append(header)
  movie_validate.append(header)
  movie_train_label.append(genres)
  movie_validate_label.append(genres)

  train = int(len(data) * 0.75)

  for i, r in data.iterrows():
    if len(str(r['release_date'])) <= 4 or len(str(r['imdb_id'])) <= 2 or len(str(r['genres'])) <= 2:
      continue
    year = r['release_date'][0:4]
    movie_id = int(r['imdb_id'][2::])

    temp_genres = re.findall(r'\w+', str(r['genres']))
    movie_genres = temp_genres[3::4]

    movie = [str(movie_id), str(year)]
    movie_genres_encoded = get_labels(genres)

    if i < train:
      movie_train.append(movie)
      movie_train_label.append(movie_genres_encoded)
    else :
      movie_validate.append(movie)
      movie_validate_label.append(movie_genres_encoded)
  write_data(movie_train, movie_train_label, movie_validate, movie_validate_label)

def main():
	genres = ['War', 'Fantasy', 'Mystery', 'TV Movie', 'Science Fiction', 'Western'
		  , 'Comedy', 'Documentary', 'Crime', 'Action', 'Music', 'Adventure', 'Family'
		  , 'Thriller', 'History', 'Horror', 'Foreign', 'Drama', 'Romance', 'Animation']
	header = ['Id', 'Year']
	create_id_label(genres, header)

if __name__ == '__main__':
    main()