import math
import numpy as np
from statistics import median
import ao_core as ao
import time
import gc
from data_prep import prepare_data


def run_ao_model(num_users:int, reviews_per_user:int, split=0.8):
    """
    function to run ao_model on a per_user level
    :param num_users: number of users you want to run the models on
    :param reviews_per_user: number of reviews each user has
    :param split: train-test split (0.8 by default) on a per-user level.
    :return:
        - average_accuracy : average accuracy of all the users that were predicted
        - median_accuracy : median of all the accuracies of all the users that were tested
        - time_taken : time taken to go through all the users
    """

    # Arch defines the architecture of our weightless neural network (wnn).
    # arch_i is the number of input neurons it should have. Remember that wnn only takes inputs in binary.
    # So all the inputs have been binary encoded as following :
    #     - 10 bit binary encoding for genres
    #     - 3 bit binary encoding for the language of the movie
    #     - 3 bit binary encoding for the average vote of the movie (from all users)
    #     - 10 bit binary encoding for number of reviews the movie has in general
    # So, there are 26 neurons for our input layer,
    # arch_z is the number of output neurons. We have chosen 10 neurons, each outputting a 1 or a 0, depending on
    # whether the movie is liked by a user or not. If more than 5 neurons are activated, we can consider that
    # the user likes the movie.
    # arch_c defines the control neurons which are responsible for pain/pleasure associations within the neural
    # architecture. There are 2 control neurons by default and so, we're not specifying any more.
    # Connector_functions define how all the neurons are connected within the wnn.
    Arch = ao.Arch(arch_i=[10, 3, 3, 10], arch_z=[10], arch_c=[], connector_function="forward_forward_conn", )

    # Calling the prepare_data function from data_prep.py code to preprocess the data for the model.
    # Users_data is a list, where each element represents a user. Each element (user) is also a list.
    # and within that list are all the encodings and labels (of that user and a random movie)
    # So, for example [[[user1, movie1, rating1],[user1, movie2, rating2]],[user2], [user3]]
    Users_data = prepare_data(reviews_per_user=reviews_per_user, num_user=num_users)
    # start recording time here
    now = time.time()

    # empty list for storing the accuracies of each user
    correct_array = []
    for index, user_data in enumerate(Users_data):
        # print("index: ", index)

        # Initializing an agent for that user with the architech we defined earlier. _steps limits the RAM usage and
        # is an optional argument
        Agent = ao.Agent(Arch, _steps=15000)

        n = len(user_data)

        # Train-test split here
        split_index= math.floor(n*split)
        train = user_data[:split_index]
        test = user_data[split_index:]
        number_test = len(test)

        # Training Phase
        inputs = []
        labels = []

        for i, row in enumerate(train):
            genres_data = row[3]
            rating_encoding = row[2]
            lang = row[4]
            vote_avg = row[5]
            vote_count = row[6]

            input_data = np.concatenate((genres_data, vote_avg,lang,vote_count))


            label = rating_encoding

            inputs.append(input_data)
            labels.append(label.tolist())

        # The Agent learns when we provide the inputs along with the labels. By specifying unsequenced=True, we're asking the Agent
        # to disregard the order in which inputs are provided (would be set False, if, say, you're doing timeseries forecasting)
        # DD stands for discriminative distance, and Hamming is for hamming distance. Since both of them are True, we'll calculate
        # the predictions using DD first, and if that fails to converge, then use Hamming distances.
        # Backprop is set to False, so Backprop_epochs aren't doing anything here
        
        Agent.next_state_batch(inputs, labels, unsequenced=True, DD=True, Hamming=True, Backprop=False, Backprop_epochs=10)

        correct = 0

        # Testing Phase
        for i, row in enumerate(test):
            genres_data = row[3]
            rating_encoding = row[2]
            lang = row[4]
            vote_avg = row[5]
            vote_count = row[6]

            input_data = np.concatenate((genres_data, vote_avg, lang, vote_count))

            for j in range(5):
                # When the label is not provided to the Agent, it predicts the output given the input
                response = Agent.next_state(input_data, unsequenced=True, DD=True, Hamming=True, Backprop=False, Backprop_type="norm")
            
            Agent.reset_state()


            distance = abs(sum(rating_encoding)-sum(response))

            if distance <= 2:
                correct += 1

        correct_array.append(correct/number_test)
        correct = 0
        Agent = None
        del(Agent)
        gc.collect()
    after = time.time()


    avg_accuracy = sum(correct_array)/len(correct_array)
    avg_median = median(correct_array)
    time_taken = after - now
    return avg_accuracy, avg_median, time_taken


if __name__=="__main__":

    accuracies = {}
    times = {}
    median_acc = {}
    num_user_list = [250]
    num_reviews_list = [5, 10, 25, 100, 200]
    for i in num_user_list:
        for j in num_reviews_list:

            acc, med, t = run_ao_model(i, j)
            print(f'accuracy for {i} num users and {j} reviews per user is {acc} and median is {med}')
            print(f'time taken was {t}')
            accuracies[str(i)+" num_users + "+str(j)+" reviews per user"] = acc
            times[str(i) + " num_users + " + str(j) + " reviews per user"] = t
            median_acc[str(i) + " num_users + " + str(j) + " reviews per user"] = med


    print(accuracies)
    print(median_acc)
    print(times)