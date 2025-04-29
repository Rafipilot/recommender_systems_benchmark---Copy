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
    - If the (`reviews_per_user` == None or 0), AND (`num_reviews` == None or 0), then we want to take all the reviews from all the users, so we jsut merge both the datasets and return final dataset
    - If (`reviews_per_user` == None or 0), AND (`num_user` != None or 0), then we want to take `reviews_per_user` number of reviews from all the users. So we merged both `movies_metadata` and `ratings` dataset, and then isolate all the users with more than `reviews_per_user` number of reviews, and then randomly sample `reviews_per_user` number of reviews from each of those users, and return the final dataset.
    - If (`reviews_per_user` != None or 0), AND (`num_user` != None or 0)
    - If (`reviews_per_user` != None or 0), AND (`num_user` == None or 0)

### [`PerUser.py`](https://github.com/saatweek/recommender_systems_benchmark/blob/main/PerUser/perUser.py)
It has 1 function, called, `run_ao_model()`, this 




   
