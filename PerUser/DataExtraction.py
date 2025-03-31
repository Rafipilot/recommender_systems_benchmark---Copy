import pandas as pd


# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  # Convert to float from object

user_counts = df2['userId'].value_counts()
selected_users = user_counts[user_counts >= 20].index  # Users with at least 20 ratings

sampled_users = selected_users[:1]  


# Get 20 ratings per sampled user
merged_df = df2[df2['userId'].isin(sampled_users)]

# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")

print(merged_df.columns)

