import pandas as pd
import ast  # Importing ast for safe evaluation

import ao_core as ao
import ao_arch as ar

# Define architecture and agent
Arch = ar.Arch(arch_i=[10, 25], arch_z=[1], arch_c=[])
Agent = ao.Agent(Arch)

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  # Convert to float from object

user_counts = df2['userId'].value_counts()
selected_users = user_counts[user_counts >= 40].index  # Users with at least 20 ratings

# Choose a subset of users (e.g., 50 users for the test)
sampled_users = selected_users[:600]  # Adjust this number as needed

# Get 20 ratings per sampled user
merged_df = df2[df2['userId'].isin(sampled_users)].groupby("userId").apply(lambda x: x.sample(40, random_state=42)).reset_index(drop=True)

# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")
print(merged_df)

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
    return format(id, '025b')

# Splitting dataset into training (80%) and testing (20%)
train_df = merged_df.sample(frac=0.8, random_state=42)
test_df = merged_df.drop(train_df.index)

# Training Phase
inputs = []
labels = []


print("Training Phase:")
for i, row in train_df.reset_index().iterrows():  
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

    input_data = genre_encoding + list(userId_encoding)
    label = rating_encoding


    inputs.append(input_data)
    labels.append(label)

Agent.next_state_batch(inputs, labels, unsequenced=True,DD=True, Hamming=False, Backprop=False, print_result=True, Backprop_epochs=10)

print("Testing Phase:")
correct = 0
for i, row in test_df.reset_index().iterrows():  
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

    input_data = genre_encoding + list(userId_encoding)

    # Instead of training, just evaluate
    response = Agent.next_state(input_data, print_result=True,DD=False, Hamming=False, Backprop=False, Backprop_type="norm", unsequenced=True)
    if response == rating_encoding:
        print("Correct!")
        correct += 1
    Agent.reset_state()
print(f"Accuracy: {correct / len(test_df)}")