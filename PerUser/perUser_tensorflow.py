import pandas as pd
import math
import ast
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available and set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# if device.type == "cuda":
#     print("GPU Name:", torch.cuda.get_device_name(0))

# Define architecture parameters for encoding (matching original feature dimensions)
start_Genre = ["drama", "comedy", "action", "romance", "documentary",
               "thriller", "adventure", "fantasy", "crime", "horror"]

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

# Convert id to numeric and filter
df1['id'] = pd.to_numeric(df1['id'], errors='coerce')
user_counts = df2['userId'].value_counts()

correct_avg_array = []

# Define encoding functions (unchanged from original logic)
def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)
    for genre in genres_list:
        if genre.lower() in start_Genre:
            genre_encoding[start_Genre.index(genre.lower())] = 1
    return genre_encoding

def encode_count(count, m):
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
    return [1] * 10 if rating >= 3.0 else [0] * 10

def encode_vote_avg(avg, count):
    # Using same thresholding logic as original
    if avg < 2:
        return [0, 0, 0]
    elif avg < 4:
        return [0, 1, 0]
    elif avg < 6:
        return [0, 1, 1]
    else:
        return [1, 1, 1]

# Define the PyTorch model (similar architecture to the Keras model)
class MovieModel(nn.Module):
    def __init__(self):
        super(MovieModel, self).__init__()
        self.fc1 = nn.Linear(18, 32)  # 18 = 10 (genre) + 3 (vote_avg) + 3 (lang) + 2 (vote_count)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Outer loop to process different user samples

overal_correct_array = []

sampled_users = user_counts.index[10000:10300]  
merged_df = df2[df2['userId'].isin(sampled_users)]
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")

# Use the 90th percentile for vote_count threshold and compute global average vote
m = merged_df['vote_count'].quantile(0.9)
C = merged_df['vote_average'].mean()
merged_df = merged_df.copy().loc[merged_df['vote_count'] >= m]

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

correct_array = []  # Accuracy for each user group

# Process each user group
now = datetime.now()
for idx, user_data in enumerate(Users_data):
    print(f"\nUser Group {idx}:")
    n = len(user_data)
    split = math.floor(n * 0.8)
    train_data = user_data[:split]
    test_data = user_data[split:]
    
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
        vote_count_encoding = encode_count(row[7], m)        # length 2
        input_vector = genre_encoding + vote_avg_encoding + lang_encoding + vote_count_encoding
        
        train_inputs.append(input_vector)
        train_labels.append(encode_rating(row[2]))           # 10-element target
    
    # Convert training data to torch tensors and send to device
    X_train = torch.tensor(np.array(train_inputs, dtype=np.float32)).to(device)
    y_train = torch.tensor(np.array(train_labels, dtype=np.float32)).to(device)
    
    # Create model, loss function and optimizer
    model = MovieModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop for 5 epochs
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        try:
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
        except Exception as e:
            print(e)

        
    
    # Testing phase for the current user group
    model.eval()
    correct = 0
    with torch.no_grad():
        for row in test_data:
            try:
                genres_data = ast.literal_eval(row[3])
                genres = [genre_dict["name"] for genre_dict in genres_data]
            except (ValueError, SyntaxError):
                genres = []
            
            genre_encoding = encode_genre(genres)
            vote_avg_encoding = encode_vote_avg(row[6], row[7])
            lang_encoding = encode_lang(row[5])
            vote_count_encoding = encode_count(row[7], m)
            input_vector = genre_encoding + vote_avg_encoding + lang_encoding + vote_count_encoding
            
            # Prepare input tensor (shape: [1, 18])
            X_test = torch.tensor(np.array([input_vector], dtype=np.float32)).to(device)
            pred = model(X_test).cpu().numpy()[0]
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

after = datetime.now()
overall_accuracy = sum(correct_array) / len(correct_array) if correct_array else 0
print(f"average accuracy: {overall_accuracy:.2f}")
print("time: ", after-now)