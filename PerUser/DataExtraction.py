import pandas as pd
import math
import ast # to convert the str to list

import ao_core as ao
import ao_arch as ar

# Define architecture and agent
Arch = ar.Arch(arch_i=[10], arch_z=[1], arch_c=[])

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  # Convert to float from object

user_counts = df2['userId'].value_counts()

sampled_users = user_counts.index[:10]  


# Get 20 ratings per sampled user
merged_df = df2[df2['userId'].isin(sampled_users)]

# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")

print(merged_df.columns)
m= merged_df['vote_count'].quantile(0.9)
print("m: ", m)
merged_df = merged_df.copy().loc[merged_df['vote_count'] >= m]
print("mergeddf: ", merged_df.shape)

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



# Training Phase
inputs = []
labels = []

sorted_merged_df = merged_df.sort_values(by=["userId"])
print("sorted_train_df: ", sorted_merged_df)




print("Training Phase:")
first_pass = True
previous_userId = None
Users_data = []  
user = []


for i, row in sorted_merged_df.reset_index().iterrows():
    if first_pass:
        first_pass = False
        la = [row["userId"], row["movieId"], row["rating"], row["genres"], row["adult"], row["original_language"], row["vote_average"], row["vote_count"]]
        user.append(la)
        previous_userId = row["userId"]
    else:
        if row["userId"] == previous_userId:
            la = [row["userId"], row["movieId"], row["rating"], row["genres"], row["adult"], row["original_language"], row["vote_average"], row["vote_count"]]
            user.append(la)
        else:
            Users_data.append(user)
            user = []

            la = [row["userId"], row["movieId"], row["rating"], row["genres"], row["adult"], row["original_language"], row["vote_average"], row["vote_count"]]
            user.append(la)
            previous_userId = row["userId"]



# Add previous user data 
if user:
    Users_data.append(user)

for index, user_data in enumerate(Users_data):
    print(f"User Group {index}:")
    print(user_data)
    print("\n") 

print(sorted_merged_df.columns)