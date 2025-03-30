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
selected_users = user_counts[user_counts >= 20].index  # Users with at least 20 ratings

sampled_users = selected_users[:200]  


# Get 20 ratings per sampled user
merged_df = df2[df2['userId'].isin(sampled_users)].groupby("userId").apply(lambda x: x.sample(20, random_state=42)).reset_index(drop=True)

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
    return format(id, '025b')

# Splitting dataset into training (80%) and testing (20%)
train_df = merged_df.sample(frac=0.8, random_state=42)
test_df = merged_df.drop(train_df.index)

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
        la = [row["userId"], row["movieId"], row["rating"], row["genres"]]
        user.append(la)
        previous_userId = row["userId"]
    else:
        if row["userId"] == previous_userId:
            la = [row["userId"], row["movieId"], row["rating"], row["genres"]]
            user.append(la)
        else:
            Users_data.append(user)
            user = []

            la = [row["userId"], row["movieId"], row["rating"], row["genres"]]
            user.append(la)
            previous_userId = row["userId"]

# Add previous user data 
if user:
    Users_data.append(user)

for index, user_data in enumerate(Users_data):
    print(f"User Group {index}:")
    print(user_data)
    print("\n") 

correct_array = []

for index, user_data in enumerate(Users_data):
    Agent = ao.Agent(Arch)

    n = len(user_data)
    split= math.floor(n*0.8)
    train = user_data[:split]
    test = user_data[split:]


    for i, row in enumerate(train):
        genres = []
        print("row: ", row)
        try:
            genres_data = ast.literal_eval(row[3]) #for some reason the genres column is a string and not a list
            for genre_dict in genres_data:
                print("genre_dict: ", genre_dict)
                genres.append(genre_dict["name"])  

        except (ValueError, SyntaxError):  
            genres = []

        rating = row[2]


        rating_encoding = encode_rating(rating)


        genre_encoding = encode_genre(genres)

        input_data = genre_encoding 
        label = rating_encoding

        inputs.append(input_data)
        labels.append(label)

    Agent.next_state_batch(inputs, labels, unsequenced=True, DD=True, Hamming=False, Backprop=False, print_result=True, Backprop_epochs=10)


    print("Testing Phase:")

    correct = 0

    for i, row in enumerate(test):
        genres = []
        print("row: ", row)
        try:
            genres_data = ast.literal_eval(row[3]) 
            for genre_dict in genres_data:
                genres.append(genre_dict["name"])  

        except (ValueError, SyntaxError):  
            genres = []

        rating = row[2]

        rating_encoding = encode_rating(rating)

        genre_encoding = encode_genre(genres)

        input_data = genre_encoding 

        response = Agent.next_state(input_data, print_result=True, DD=False, Hamming=False, Backprop=False, Backprop_type="norm", unsequenced=True)
        Agent.reset_state()
        if response == rating_encoding:
            print("Correct!")
            correct += 1
    
    n = len(test)
    print("test:")
    print(test)
    print("len(test): ", len(test))


    print("correct: ", correct)
    print("correct / n: ", correct/n)

    correct_array.append(correct/n)

print("correct_array: ", correct_array)
print("average correct: ", sum(correct_array)/len(correct_array))
