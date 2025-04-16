import math
import numpy as np # to convert the str to list

import ao_core as ao

from datetime import datetime
import gc 

from data_prep import prepare_data


def run_ao_model(num_users, reviews_per_user, Arch, split=0.8):
    Users_data = prepare_data(reviews_per_user=reviews_per_user, num_user=num_users)
    now = datetime.now()
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
    after = datetime.now()


    avg_accuracy = sum(correct_array)/len(correct_array)
    time_taken = after - now
    return avg_accuracy, time_taken


if __name__=="__main__":
    Arch = ao.Arch(arch_i=[10, 3, 3, 10], arch_z=[10], arch_c=[], connector_function="forward_forward_conn", )
    accuracies = {}
    times = {}
    num_user_list = [100]#, 500, 1000]
    num_reviews_list = [50]#, 200, 500, 1000]
    for i in num_user_list:
        for j in num_reviews_list:
            try :
                acc, t = run_ao_model(i, j, Arch )
                print(f'accuracy for {i} num users and {j} reviews per user is {acc}')
                print(f'time taken was {t}')
                accuracies[str(i)+" num_users + "+str(j)+" reviews per user"] = acc
                times[str(i) + " num_users + " + str(j) + " reviews per user"] = t
            except:
                pass

    print(accuracies)
    print(times)