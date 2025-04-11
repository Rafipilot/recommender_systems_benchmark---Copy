
import math


import ao_core as ao
import ao_arch as ar
from datetime import datetime
import gc 

from sklearn.model_selection import train_test_split



from data_prep import prepare_data


number_examples = 300
split= 0.8


def test_train():
    correct_array = []
    for index, user_data in data_set.iterrows():
        print("index: ", index)
        Agent = ao.Agent(Arch, _steps=15000)

        n = len(user_data)
        split= math.floor(n*split)
        train, test = train_test_split(user_data, split=0.8)


        # Training Phase
        inputs = []
        labels = []



        print("len train: ", len(train))
        for i, row in enumerate(train):

            genres_data = row["genres_enc"] 

            rating = row["target"]


            lang = row["lang_enc"]

            vote_avg = row["vote_avg_enc"]
            vote_count = row["vote_count_enc"]

            input_data = genres_data  +  vote_avg + lang + vote_count
            
            label = rating



            inputs.append(input_data)
            labels.append(label)

        try:
            Agent.next_state_batch(inputs, labels, unsequenced=True, DD=True, Hamming=True, Backprop=False, Backprop_epochs=10)

        except Exception as e:
            print(e)

        correct = 0


        print("len test: ", len(test))
        for i, row in enumerate(test):

            genres_data =row["genres_enc"]

            rating = row["target"]


            lang = row["lang_enc"]

            vote_avg = row["vote_avg_enc"]
            vote_count = row["vote_count_enc"]

            input_data = genres_data  +  vote_avg + lang + vote_count

            for j in range(5):
                response = Agent.next_state(input_data,  DD=True, Hamming=True, Backprop=False, Backprop_type="norm")
            
            Agent.reset_state()


            ones = sum(response)
            if ones >=5:
                response = 1
            else:
                response = 0
            
            ones_2 = sum(rating)
            if ones_2 >=5:
                rating = 1
            else:
                rating = 0
            

            if response == rating:
                correct += 1

            if rating == 1:
                rating = 10*[1]
            else:
                rating = 10*[0]
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

data_set = prepare_data(reviews_per_user=100, training_ratio=None)                                                                                                                                              
print(data_set.head())
# Add previous user data 


now = datetime.now()
# correct_array = test_train()
after = datetime.now()
# print("correct_array: ", correct_array)
# print("average correct: ", sum(correct_array)/len(correct_array))
# print("time: ", after - now)

# 8.12.sleep(3)


# plt.plot(avg_correct_array)
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()
