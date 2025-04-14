import math
import numpy as np # to convert the str to list

import ao_core as ao
import ao_arch as ar
from datetime import datetime
import gc 

from data_prep import prepare_data


def test_train(split= 0.8):
    correct_array = []
    for index, user_data in enumerate(Users_data):
        print("index: ", index)
        Agent = ao.Agent(Arch, _steps=15000)

        n = len(user_data)
        print("len user data: ", len(user_data))
        split= math.floor(n*0.8)
        print("split: ", split)
        train = user_data[:split]
        test = user_data[split:]
        number_test = len(test)

        # Training Phase
        inputs = []
        labels = []



        print("len train: ", len(train))
        for i, row in enumerate(train):


            genres_data = row[3]
    

            rating_encoding = row[2]

            lang = row[4]


            vote_avg = row[5]
            vote_count = row[6]




            input_data = np.concatenate((genres_data, vote_avg,lang,vote_count))


            if rating_encoding == 1:
                rating_encoding = 10*[1]
            else:
                rating_encoding = 10*[0]
            

            label = rating_encoding




            inputs.append(input_data)
            labels.append(label)



        Agent.next_state_batch(inputs, labels, unsequenced=True, DD=True, Hamming=True, Backprop=False, Backprop_epochs=10)



        correct = 0


        print("len test: ", len(test))
        for i, row in enumerate(test):
                
            genres_data = row[3]
    

            rating_encoding = row[2]

            lang = row[4]


            vote_avg = row[5]
            vote_count = row[6]

            input_data = np.concatenate((genres_data, vote_avg,lang,vote_count))

            for j in range(5):
                response = Agent.next_state(input_data,  DD=True, Hamming=True, Backprop=False, Backprop_type="norm")
            
            Agent.reset_state()



            
            if rating_encoding == 1:
                rating_encoding = 10*[1]
            else:
                rating_encoding = 10*[0]


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

            #Agent.next_state(input_data, rating_encoding, DD=False, Hamming=False, Backprop=False, Backprop_type="norm", unsequenced=True)
            #Agent.reset_state()

        correct_array.append(correct/number_test)
        correct = 0
        Agent = None
        del(Agent)
        gc.collect()
    return correct_array


# Define architecture and agent
Arch = ar.Arch(arch_i=[10, 3, 3, 10], arch_z=[10], arch_c=[], connector_function="forward_forward_conn",)




Users_data = prepare_data(reviews_per_user=100, num_user=10)
print(Users_data[0][:5])

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