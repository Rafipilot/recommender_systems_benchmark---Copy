import pandas as pd
import math
import ast
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define architecture parameters for encoding (matching original feature dimensions)
start_Genre = ["drama", "comedy", "action", "romance", "documentary",
               "thriller", "adventure", "fantasy", "crime", "horror"]

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

# Convert id to numeric and filter
df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
user_counts = df2['userId'].value_counts()
sampled_users = user_counts.index[1000:1010]
merged_df = df2[df2['userId'].isin(sampled_users)]
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")

# Use the 90th percentile for vote_count threshold and compute global average vote
m = merged_df['vote_count'].quantile(0.9)
C = merged_df['vote_average'].mean()
merged_df = merged_df.copy().loc[merged_df['vote_count'] >= m]

# Encoding functions (unchanged)
def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)
    for genre in genres_list:
        if genre.lower() in start_Genre:
            genre_encoding[start_Genre.index(genre.lower())] = 1
    return genre_encoding

def encode_count(count):
    if count < m + 200:
        return [0, 0]
    elif count < m + 600:
        return [0, 1]
    else:
        return [1, 1]

def encode_lang(lang):
    if lang == "en":
        return [0, 0, 0]
    elif lang == "fr":
        return [0, 0, 1]
    elif lang == "it":
        return [0, 1, 1]
    elif lang == "ja":
        return [1, 1, 1]
    elif lang == "de":
        return [1, 0, 0]
    else:
        return [1, 1, 0]

def encode_rating(rating):
    # Returns a 10-element vector for consistency with the original encoding
    return 10 * [1] if rating >= 3.0 else 10 * [0]

def encode_adult(adult):
    return [1] if adult == True else [0]

def encode_vote_avg(avg, count):
    # Prints are omitted in Keras version; using same thresholding logic.
    if avg < 2:
        return [0, 0, 0]
    elif avg < 4:
        return [0, 1, 0]
    elif avg < 6:
        return [0, 1, 1]
    else:
        return [1, 1, 1]

# Sort merged data by userId and group data per user
sorted_merged_df = merged_df.sort_values(by=["userId"])
Users_data = []
current_user = []

previous_userId = None
for _, row in sorted_merged_df.iterrows():
    row_data = [row["userId"], row["movieId"], row["rating"],
                row["genres"], row["adult"], row["original_language"],
                row["vote_average"], row["vote_count"]]
    if previous_userId is None or row["userId"] == previous_userId:
        current_user.append(row_data)
    else:
        Users_data.append(current_user)
        current_user = [row_data]
    previous_userId = row["userId"]
if current_user:
    Users_data.append(current_user)

# Create a Keras model for benchmark
def create_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(18,)),  # 18 = 10 (genre) + 3 (vote_avg) + 3 (lang) + 2 (vote_count)
        layers.Dense(16, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Benchmark across each user group
correct_array = []

for idx, user_data in enumerate(Users_data):
    print(f"\nUser Group {idx}:")
    
    n = len(user_data)
    split = math.floor(n * 0.8)
    train_data = user_data[:split]
    test_data = user_data[split:]
    
    # Prepare training data lists
    train_inputs = []
    train_labels = []
    
    for row in train_data:
        # Parse genres (if possible, otherwise empty list)
        try:
            genres_data = ast.literal_eval(row[3])
            genres = [genre_dict["name"] for genre_dict in genres_data]
        except (ValueError, SyntaxError):
            genres = []
            
        # Build input features (concatenate the encodings)
        genre_encoding = encode_genre(genres)              # length 10
        vote_avg_encoding = encode_vote_avg(row[6], row[7])    # length 3
        lang_encoding = encode_lang(row[5])                  # length 3
        vote_count_encoding = encode_count(row[7])           # length 2
        input_vector = genre_encoding + vote_avg_encoding + lang_encoding + vote_count_encoding
        
        train_inputs.append(input_vector)
        train_labels.append(encode_rating(row[2]))           # 10-element target
    
    # Convert to numpy arrays
    X_train = np.array(train_inputs, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.float32)
    
    model = create_model()
    print("Training model on user group data...")
    model.fit(X_train, y_train, epochs=10, verbose=1)
    
    # Testing phase
    correct = 0
    for row in test_data:
        try:
            genres_data = ast.literal_eval(row[3])
            genres = [genre_dict["name"] for genre_dict in genres_data]
        except (ValueError, SyntaxError):
            genres = []
        
        genre_encoding = encode_genre(genres)
        vote_avg_encoding = encode_vote_avg(row[6], row[7])
        lang_encoding = encode_lang(row[5])
        vote_count_encoding = encode_count(row[7])
        input_vector = genre_encoding + vote_avg_encoding + lang_encoding + vote_count_encoding
        X_test = np.array([input_vector], dtype=np.float32)
        
        # Get prediction; average over the 10 neurons (simulate the summing and thresholding)
        pred = model.predict(X_test, verbose=0)[0]
        pred_sum = np.sum(pred >= 0.5)  # count number of neurons above threshold 0.5
        predicted_label = 1 if pred_sum >= 5 else 0
        
        # Ground truth: based on the rating encoding threshold
        gt_sum = np.sum(encode_rating(row[2]))
        true_label = 1 if gt_sum >= 5 else 0
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / len(test_data) if test_data else 0
    correct_array.append(accuracy)
    print(f"User group {idx} accuracy: {accuracy:.2f}")

overall_accuracy = sum(correct_array) / len(correct_array) if correct_array else 0
print(f"\nOverall average accuracy: {overall_accuracy:.2f}")
