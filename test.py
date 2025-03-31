import pandas as pd
import ast
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  # Convert to numeric

# Get 40 ratings per sampled user
merged_df = df2.reset_index(drop=True)

# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")

# Define genre categories
start_Genre = ["drama", "comedy", "action", "romance", "documentary", "thriller", "adventure", "fantasy", "crime", "horror"]

# Encoding functions
def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)  
    for genre in genres_list:
        if genre.lower() in start_Genre:
            genre_encoding[start_Genre.index(genre.lower())] = 1
    return genre_encoding

def encode_rating(rating):
    return [1] if rating >= 3.0 else [0]

def encode_Id(id):
    return [int(bit) for bit in format(id, '025b')]  # Convert to binary and list

# Prepare dataset
inputs = []
labels = []

for _, row in merged_df.iterrows():  
    genres = []
    try:
        genres_data = ast.literal_eval(row["genres"]) 
        for genre_dict in genres_data:
            genres.append(genre_dict["name"]) 
    except (ValueError, SyntaxError):  
        genres = []

    rating = row["rating"]
    userId = row["userId"]

    rating_encoding = encode_rating(rating)
    userId_encoding = encode_Id(userId)
    genre_encoding = encode_genre(genres)

    input_data = genre_encoding + userId_encoding
    label = rating_encoding

    inputs.append(input_data)
    labels.append(label)

# Convert to NumPy arrays
X = np.array(inputs, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Keras model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),  # Input layer
    layers.Dense(32, activation="relu"),  # Hidden layer
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Output layer (Binary classification)
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Evaluate accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
