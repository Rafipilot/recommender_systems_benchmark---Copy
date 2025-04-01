import pandas as pd
import math
import ast # to convert the str to list

import ao_core as ao
import ao_arch as ar

# Define architecture and agent
Arch = ar.Arch(arch_i=[10, 3, 3], arch_z=[10], arch_c=[])

# Load datasets
df1 = pd.read_csv("data/movies_metadata.csv")
df2 = pd.read_csv("data/ratings.csv")

df1['id'] = pd.to_numeric(df1['id'], errors='coerce')  # Convert to float from object

user_counts = df2['userId'].value_counts()

sampled_users = user_counts.index[:50]  

merged_df = df2[df2['userId'].isin(sampled_users)]

# Merge with movie metadata
merged_df = merged_df.merge(df1, left_on="movieId", right_on="id", how="inner")


m= merged_df['vote_count'].quantile(0.9)
C= merged_df['vote_average'].mean()
merged_df = merged_df.copy().loc[merged_df['vote_count'] >= m]  # Remove bottom 90


# Define genre categories
start_Genre = ["drama", "comedy", "action", "romance", "documentary", "thriller", "adventure", "fantasy", "crime", "horror"]

# Encoding functions
def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)  
    for genre in genres_list:
        if genre.lower() in start_Genre:
            genre_encoding[start_Genre.index(genre.lower())] = 1

    return genre_encoding

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
    v = avg
    R = count
    # Calculation based on the IMDB formula
    avg =  (v/(v+m) * R) + (m/(m+v) * C)
    if avg < 4:
        return [0,0,0]
    elif avg <8:
        return [0,1,0]
    elif avg<12:
        return [0,1,1]
    else:
        return [1,1,1]


# Training Phase
inputs = []
labels = []

sorted_merged_df = merged_df.sort_values(by=["userId"])


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

        


        rating_encoding = encode_rating(rating)

        genre_encoding = encode_genre(genres)

        input_data = genre_encoding  +  vote_avg_encoding + lang_encoding

        print("input: ", input_data)
        
        label = rating_encoding
        print("label: ", label)

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

        adult = row[4]
        adult_encoding = encode_adult(adult)

        lang = row[5]
        lang_encoding = encode_lang(lang)

        vote_avg = row[6]
        vote_count = row[7]
        vote_avg_encoding = encode_vote_avg(vote_avg, vote_count)

        


        rating_encoding = encode_rating(rating)

        genre_encoding = encode_genre(genres)

        input_data = genre_encoding  +  vote_avg_encoding + lang_encoding

        for i in range(5):
            response = Agent.next_state(input_data, print_result=True, DD=True, Hamming=False, Backprop=False, Backprop_type="norm", unsequenced=True)
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
        Agent.reset_state()
        print(rating_encoding)
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

    correct = 0

print("correct_array: ", correct_array)
print("average correct: ", sum(correct_array)/len(correct_array))
