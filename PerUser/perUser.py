import pandas as pd
import math
import ast # to convert the str to list

import ao_core as ao
import ao_arch as ar
from datetime import datetime
import gc 
import matplotlib.pyplot as plt

from data_prep import prepare_data


def test_train(number_examples = 300, number_reviews=None, split= 0.8, top_percentile=100):
    correct_array = []
    for index, user_data in enumerate(Users_data[:number_examples]):
        print("index: ", index)
        Agent = ao.Agent(Arch, _steps=15000)

        n = len(user_data)
        split= math.floor(n*split)
        train = user_data[:split]
        test = user_data[split:]

        # Training Phase
        inputs = []
        labels = []



        print("len train: ", len(train))
        for i, row in enumerate(train):
            genres = []
            try:
                genres_data = ast.literal_eval(row[3]) #for some reason the genres column is a string and not a list
                for genre_dict in genres_data:
                    genres.append(genre_dict["name"])  

            except (ValueError, SyntaxError):  
                genres = []

            rating = row[2]

            adult = row[4]
            adult_encoding = encode_adult(adult)

            lang = row[5]
            lang_encoding = encode_lang(lang)

            vote_avg = row[6]
            vote_count = row[7]
            vote_avg_encoding = encode_vote_avg(vote_avg, vote_count)
            vote_count_encoding = encode_count(vote_count)

            


            rating_encoding = encode_rating(rating)

            genre_encoding = encode_genre(genres)

            input_data = genre_encoding  +  vote_avg_encoding + lang_encoding + vote_count_encoding
            
            label = rating_encoding




            inputs.append(input_data)
            labels.append(label)

        try:
            Agent.next_state_batch(inputs, labels, unsequenced=True, DD=True, Hamming=True, Backprop=False, Backprop_epochs=10)

        except Exception as e:
            print(e)

        correct = 0


        print("len test: ", len(test))
        for i, row in enumerate(test):

            genres = []
            try:
                genres_data = ast.literal_eval(row[3]) 
                for genre_dict in genres_data:
                    genres.append(genre_dict["name"])  

            except (ValueError, SyntaxError):  
                genres = []


            rating = row[2]

            adult = row[4]
            adult_encoding = encode_adult(adult)

            lang = row[5]
            lang_encoding = encode_lang(lang)

            vote_avg = row[6]
            vote_count = row[7]
            vote_avg_encoding = encode_vote_avg(vote_avg, vote_count)
            vote_count_encoding = encode_count(vote_count)

            


            rating_encoding = encode_rating(rating)

            genre_encoding = encode_genre(genres)

            input_data = genre_encoding  +  vote_avg_encoding + lang_encoding + vote_count_encoding

            for j in range(5):
                response = Agent.next_state(input_data,  DD=True, Hamming=True, Backprop=False, Backprop_type="norm")
            
            Agent.reset_state()


            ones = sum(response)
            if ones >=5:
                response = 1
            else:
                response = 0
            
            ones_2 = sum(rating_encoding)
            if ones_2 >=5:
                rating_encoding = 1
            else:
                rating_encoding = 0
            

            if response == rating_encoding:
                correct += 1

            if rating_encoding == 1:
                rating_encoding = 10*[1]
            else:
                rating_encoding = 10*[0]
            #Agent.next_state(input_data, rating_encoding, DD=False, Hamming=False, Backprop=False, Backprop_type="norm", unsequenced=True)
            #Agent.reset_state()

        number_test = (1-split) * number_examples
        correct_array.append(correct/number_test)
        correct = 0
        Agent = None
        del(Agent)
        gc.collect()
    return correct_array


# Define architecture and agent
Arch = ar.Arch(arch_i=[10, 3, 3, 2], arch_z=[10], arch_c=[], connector_function="forward_forward_conn",)

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  

user_counts = df2['userId'].value_counts()  ## ordering the array by the amount of times thats users ID is present 

sampled_users = user_counts.index[0:50]  ## sampling 10k - 10.3 k users

merged_df = df2[df2['userId'].isin(sampled_users)] ## filtering df to only include the users we sampled abov



# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")


m= merged_df['vote_count'].quantile(0.9)
C= merged_df['vote_average'].mean()

# Define genre categories
start_Genre = ["drama", "comedy", "action", "romance", "documentary", "thriller", "adventure", "fantasy", "crime", "horror"]

# Encoding functions
def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)  
    for genre in genres_list:
        if genre.lower() in start_Genre:
            genre_encoding[start_Genre.index(genre.lower())] = 1

    return genre_encoding

def encode_count(count):
    if count < m+200:
        return [0,0]
    elif count < m+600:
        return [0,1]
    else:
        return [1,1]

def encode_lang(lang):

    if lang == "en":
        return [0,0,0]
    elif lang == "fr":
        return [0,0,1]
    elif lang == "it":
        return [0,1,1]
    elif lang =="ja":
        return [1, 1, 1]
    elif lang == "de":
        return [1,0,0]
    else:
        return [1, 1, 0]


def encode_rating(rating):
    return 10*[1] if rating >= 3.0 else 10*[0]

def encode_adult(adult):
    return [1] if adult==True else [0]

def encode_vote_avg(avg, count):
    
    if avg < 2:
        return [0,0,0]
    elif avg <4:
        return [0,1,0]
    elif avg<6:
        return [0,1,1]
    else:
        return [1,1,1]

sorted_merged_df = merged_df.sort_values(by=["userId"])

#get 1000 reviuws per user



User_data = prepare_data(reviews_per_user=100)

now = datetime.now()
correct_array = test_train()
after = datetime.now()
print("correct_array: ", correct_array)
print("average correct: ", sum(correct_array)/len(correct_array))
print("time: ", after - now)

# 8.12.sleep(3)


# plt.plot(avg_correct_array)
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()