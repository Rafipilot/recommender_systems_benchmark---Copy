# Recommender Systems Benchmark

## Introduction

Given that conventional recommenders, while deeply effective, rely on large distributed systems pre-trained on aggregate user data, incorporating new data necessitates large training cycles, 
making them slow to adapt to real-time user feedback and often lacking transparency in recommendation rationale. We
explore the performance of smaller personal models trained on per-user data using Weightless Neural
Networks (WNNs), an alternative to neural backpropagation that enable continuous learning by using
neural networks as a state machine rather than a system with pretrained weights. We contrast our
approach against a classic weighted system (also on a per-user level), and the industry standard,
collaborative filtering, achieving similar levels of accuracy on MovieLens dataset. 

## Installation Guide

`pip install -r requirements.txt`

## File Structure and Functions

This program is consists of 2 broad categories:
 - Per-User level recommendation code ([`PerUser`](https://github.com/saatweek/recommender_systems_benchmark/tree/main/PerUser) folder)
   - [Per User recommendation using AO Lab's weightless neural networks](https://github.com/saatweek/recommender_systems_benchmark/blob/main/PerUser/perUser.py)
   - [Per User recommendation using a typical neural network](https://github.com/saatweek/recommender_systems_benchmark/blob/main/PerUser/perUser_pytorch.py)
 - [Collaborative Filtering codes](https://github.com/saatweek/recommender_systems_benchmark/blob/main/Collaborative/torch_colab.py) ([`Collaborative`](https://github.com/saatweek/recommender_systems_benchmark/tree/main/Collaborative) folder)

Along with these codes for the different models, there are 2 additional programs : 
-  [To download and preprocess the data that we need for our models](https://github.com/saatweek/recommender_systems_benchmark/blob/main/data_prep.py)
-  [To compare all the models and save the results in a .csv file.](https://github.com/saatweek/recommender_systems_benchmark/blob/main/compare_all.py)

## Code Breakdown
A line-by-line breakdown of most of the important lines of code can be found in the comments of the respective file the function belongs to. This section
goes over what each file does overall, and describes a few important functions

### [`data_prep.py`](https://github.com/saatweek/recommender_systems_benchmark/blob/main/data_prep.py)

- We first define all the fuctions to binary encode the columns we are interested in, these are `encode_genres()` : to encode the genres the movie belongs to,
`encode_lang()` : to encode the original language the movie was released in, `encode_vote_count()` : to encode the number of reviews the movie
has in general, and finally, `encode_vote_average`: to encode the overall rating the movie has

- We use binary encodings for encoding all our input columns because weightless neural netowrks only take binary inputs, and then only output binary values
- We then define the `data_prep()` function that prepares the data on which we train all the 3 models (weightless neural networks, weighted neural networks, and collaborative filtering)
  - We first use `kagglehub` to download the data, if the data is already downloaded then it just uses the already saved data.
  - We then load 2 datasets from the movies dataset, the `movies_metadata` : which has all the information about all the movies (actors, directors, language, genre etc) and `ratings` dataset: which has information about all the ratings given by each user to whatever movie they've watched.
  - We handle all the missing values in `movies_metadata` and filter all the popular movies if `top_percentile` is specified in the `data_prep` parameter.
  - There are 2 important parameters that we need to specificy in the function : `num_user` and `reviews_per_user`. `num_user` gives us the number of unique users we want in our final dataset, and `reviews_per_user` specifies how many reviews should each of those unique users should have. Depending on the values of both, there are 4 possible cases :
    - If the (`reviews_per_user` == None or 0), AND (`num_reviews` == None or 0), then we'll take all the reviews from all the users
    - If (`reviews_per_user` == None or 0), AND (`num_user` != None or 0), then we'll take all the reviews from `num_user` randomly sampled users. 
    - If (`reviews_per_user` != None or 0), AND (`num_user` != None or 0), then we'll sample `reviews_per_user` reviews from `num_user` randomly sampled users. 
    - If (`reviews_per_user` != None or 0), AND (`num_user` == None or 0), then we'll sample `reviews_per_user` reviews from all the users with >= `reviews_per_user` reviews
  - Once we filter the number of users and the number of reviews for each user, we then binary encode all the necessary columns according to the functions we defined earlier. 
  - If the `per_user` parameter is False, we'll return the merged data after the previous processing
  - If the `per_user` parameter is True, then Uers_data is returned. Users_data is a list, where each element represents a user. Each element (user) is also a list. And within that list are all the encodings and labels (of that user and a random movie. So, for example [[[user1, movie1, rating1],[user1, movie2, rating2]],[user2], [user3]]

### [`PerUser.py`](https://github.com/saatweek/recommender_systems_benchmark/blob/main/PerUser/perUser.py)
It has 1 function, called, `run_ao_model()`, this function takes in  `num_users` and `reviews_per_user` and `split`. 
- `num_users` takes in an integer and is the number of users you want to run the model for (or number of individual models you want to test). If 0 or None, then all the users are taken
- `reviews_per_user` takes in an integer and is the number of reviews of each user you want to consider. If 0 or None, then all the reviews are considered to train/test the model
- `split` is the ratio of training set from the entire dataset. The default value is 0.8, therefore 80-20 split is considered for training and testing.
- The function returns 3 values :
   - the mean accuracy of all the users
   - the median accuracy of all the users
   - Time taken for the model to go through all the users. 

### [`perUser_pytorch.py`](https://github.com/saatweek/recommender_systems_benchmark/blob/one_hot/PerUser/perUser_pytorch.py)
It also has just one function with exactly the same parameters as `perUser.py` which do the exact same thing and return the exact same values

### [`torch_colab`](https://github.com/saatweek/recommender_systems_benchmark/blob/one_hot/Collaborative/torch_colab.py)
It also has only one function, but with 2 arguments. `num_users` and `reviews_per_user`. And unlike the other two functions, it only returns average accuracy and the time taken by the model to train and test. 

### [`compare_all.py`](https://github.com/saatweek/recommender_systems_benchmark/blob/one_hot/compare_all.py)
Program to run all the models and save the results to a .csv file. 
