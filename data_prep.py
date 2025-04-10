import kagglehub
import pandas as pd
import numpy as np
import os
import ast
from sklearn import model_selection, preprocessing
import math


def encode_genres(s: str) -> np.ndarray:
    """
    encodes genres into predefined bins
    :param s: the 'genres' column of the data has a string whcih is a dictionary of all the genres the movie belongs to
    :return: encoded genre
    """
    start_genres = ["drama", "comedy", "action", "romance", "documentary",
              "thriller", "adventure", "fantasy", "crime", "horror"]
    genre_list = []
    try:
        genres = ast.literal_eval(s)
        for item in genres:
            genre_list.append(item['name'].lower())
    except:
        pass

    encoding = [0] * len(start_genres)
    for genre in genre_list:
        if genre in start_genres:
            encoding[start_genres.index(genre)] = 1
    return np.array(encoding)

def encode_lang(lang: str) -> np.ndarray:
    """
    encodes the language of the movie. The highest occurring languages have predefined encodings, the rest are encoded as [1,1,1]
    :param lang: the "original_language" column from the dataframe.
    :return: numpy array of encodings
    """
    lang = lang.lower()
    if lang == "en":
        return np.array([0, 0, 0])
    elif lang == "fr":
        return np.array([0, 0, 1])
    elif lang == "de":
        return np.array([0, 1, 0])
    elif lang == "ja":
        return np.array([0, 1, 1])
    elif lang == "it":
        return np.array([1, 0, 0])
    elif lang == "es":
        return np.array([1, 1, 0])
    else:
        return np.array([1, 1, 1])

def encode_vote_count(num: int) -> np.ndarray:
    """
    encoding vote_count by putting it into bins of 100 (so, movies less than 100 go in first bin,
    movies with 100-200 reviews go in second bin and so on
    :param num: int of the number of vote_counts
    :return: numpy array of bins, 1 indicating which bin the number belongs to
    """
    bins = [0] * 10
    try:
        idx = min(math.floor(num / 100), 9)
        bins[idx] = 1
    except:
        bins[-1] = 1
    return np.array(bins)

def encode_vote_avg(avg: float) -> np.ndarray:
    """
    vote_average is a float from 0-5, this function puts them into bins accordingly
    :param avg: vote_average
    :return: binned value of vote_average
    """
    if avg < 1:
        return np.array([0, 0, 0])
    elif avg < 2:
        return np.array([0, 0, 1])
    elif avg < 3:
        return np.array([0, 1, 0])
    elif avg < 4:
        return np.array([0, 1, 1])
    else:
        return np.array([1, 1, 1])


def prepare_data(reviews_per_user:int | None = None,
                 training_ratio:float | None = 0.8,
                 num_examples:int= 10000,
                 top_percentile:float | None = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares the data to be input for different ML models
    :param reviews_per_user: number of reviews you want from each user in the final database
    :param training_ratio: if specified, train:test ratio, else the entire dataset as is
    :param num_examples: number of examples in the final dataset (before splitting into train-test)
    :param top_percentile: if specified, only movies with vote_count in the top_percentile would be considered for training
    :return:
    """
    # Download dataset
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

    # Load data
    movies_metadata = pd.read_csv(os.path.join(path, "movies_metadata.csv"), low_memory=False)
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))

    # Converting ids to numeric
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')

    # Handling missing values
    movies_metadata = movies_metadata.dropna(subset=['id'])
    movies_metadata['genres'] = movies_metadata['genres'].fillna('[]')
    movies_metadata['original_language'] = movies_metadata['original_language'].fillna('en')
    movies_metadata['vote_count'] = movies_metadata['vote_count'].fillna(0).astype(int)
    movies_metadata['vote_average'] = movies_metadata['vote_average'].fillna(0)

    if top_percentile is not None:
        m = ratings['vote_count'].quantile(top_percentile)
        C = ratings['vote_average'].mean()
        movies_metadata = movies_metadata[movies_metadata['vote_average'] >= m]

    # Sorting rows according to time, and deleting the duplicate rows keeping only the last occurrence
    ratings.sort_values('timestamp', inplace=True)
    ratings = ratings.drop_duplicates(['userId', 'movieId'], keep='last')


    merged = ratings.merge(movies_metadata, left_on='movieId', right_on='id', how='inner')

    if reviews_per_user is not None:
        user_review_counts = merged.groupby('userId')['rating'].count()
        heavy_users = user_review_counts[user_review_counts >= reviews_per_user].index
        merged = merged[merged['userId'].isin(heavy_users)].groupby('userId').sample(n=reviews_per_user, random_state=9)

    merged = merged.sample(n=min(num_examples, len(merged)), random_state=99)

    merged['genres_enc'] = merged['genres'].apply(encode_genres)
    merged['lang_enc'] = merged['original_language'].apply(encode_lang)
    merged['vote_count_enc'] = merged['vote_count'].apply(encode_vote_count)
    merged['vote_avg_enc'] = merged['vote_average'].apply(encode_vote_avg)
    merged['target'] = (merged['rating'] >= 3).astype(int)  # Binary classification target

    # Label encoding
    user_encoder = preprocessing.LabelEncoder()
    movie_encoder = preprocessing.LabelEncoder()
    merged['user_id'] = user_encoder.fit_transform(merged['userId'])
    merged['movie_id'] = movie_encoder.fit_transform(merged['movieId'])

    if training_ratio is not None:
        train_df, val_df = model_selection.train_test_split(merged, train_size=training_ratio, stratify=merged['target'], random_state=999)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    else:
        return merged