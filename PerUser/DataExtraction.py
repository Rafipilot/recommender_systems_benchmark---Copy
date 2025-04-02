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



m= merged_df['vote_count'].quantile(0.9)
merged_df = merged_df.copy().loc[merged_df['vote_count'] >= m]
C= merged_df['vote_average'].mean()
n_mean = merged_df['vote_count'].mean()
print(n_mean)
print("c: ", C)
print("m: ", m)

print("mergeddf: ", merged_df.shape)
print(merged_df.index)


merged_df.filter(["used_Id"])

