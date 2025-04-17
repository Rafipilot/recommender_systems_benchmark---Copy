from Collaborative.torch_colab import run_colab_model
from PerUser.perUser_pytorch import run_torch_per_user
from PerUser.perUser import run_ao_model
import pandas as pd
import os

data = {'num_users':[100, 100, 100, 100, 500, 500, 500, 1000, 1000, 1000], 'reviews_per_user':[50, 200, 500, 1000, 50, 200, 500, 50, 200, 500]}
df = pd.DataFrame(data)
for iteration in range(5):
    ao_acc_list = []
    ao_med_list = []
    ao_time_list = []
    torch_acc_list = []
    torch_med_list = []
    torch_time_list = []
    colab_acc_list = []
    colab_time_list = []
    for idx, items in df.iterrows():
        print(f'Running ao model for num_users={items.iloc[0]} with {items.iloc[1]} reviews per user..')
        ao_acc, ao_med, ao_t = run_ao_model(num_users=items.iloc[0], reviews_per_user=items.iloc[1])
        ao_acc_list.append(ao_acc)
        ao_med_list.append(ao_med)
        ao_time_list.append(ao_t)
        print(f'Running torch model...')
        torch_acc, torch_med, torch_t = run_torch_per_user(num_users=items.iloc[0], reviews_per_user=items.iloc[1])
        torch_acc_list.append(torch_acc)
        torch_med_list.append(torch_med)
        torch_time_list.append(torch_t)
        print(f'Running collaborative model...')
        colab_acc, colab_time = run_colab_model(num_users=items.iloc[0], reviews_per_user=items.iloc[1])
        colab_acc_list.append(colab_acc)
        colab_time_list.append(colab_time)

    df['iter_' + str(iteration) + '_ao_avg_accuracy'] = ao_acc_list
    df['iter_' + str(iteration) + '_ao_median_accuracy'] = ao_med_list
    df['iter_' + str(iteration) + '_ao_time'] = ao_time_list
    df['iter_' + str(iteration) + '_torch_avg_accuracy'] = torch_acc_list
    df['iter_' + str(iteration) + '_torch_median_accuracy'] = torch_med_list
    df['iter_' + str(iteration) + '_torch_time'] = torch_time_list
    df['iter_' + str(iteration) + '_colab_accuracy'] = colab_acc_list
    df['iter_' + str(iteration) + '_colab_time'] = colab_time_list


df.to_csv('final_results.csv', index=False)

