import time
from Collaborative.torch_colab import run_colab_model
from PerUser.perUser_pytorch import run_torch_per_user
from PerUser.perUser import run_ao_model
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = {'num_users':[100, 100, 100, 100], 'reviews_per_user':[5, 10, 15, 25]}
df = pd.DataFrame(data)
df = df.fillna(0)
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
        print(f'(iteration #{iteration+1})Running ao model for num_users={items.iloc[0]} with {items.iloc[1]} reviews per user..')
        time.sleep(181)
        ao_acc, ao_med, ao_t = run_ao_model(num_users=int(items.iloc[0]), reviews_per_user=int(items.iloc[1]))
        ao_acc_list.append(ao_acc)
        ao_med_list.append(ao_med)
        ao_time_list.append(ao_t)
        print(f'(iteration #{iteration+1})Running torch model...')
        time.sleep(181)
        torch_acc, torch_med, torch_t = run_torch_per_user(num_users=int(items.iloc[0]), reviews_per_user=int(items.iloc[1]))
        torch_acc_list.append(torch_acc)
        torch_med_list.append(torch_med)
        torch_time_list.append(torch_t)
        print(f'(iteration #{iteration+1})Running collaborative model...')
        time.sleep(181)
        colab_acc, colab_time = run_colab_model(num_users=int(items.iloc[0]), reviews_per_user=int(items.iloc[1]))
        colab_acc_list.append(colab_acc)
        colab_time_list.append(colab_time)

    df['iter_' + str(iteration+1) + '_ao_avg_accuracy'] = ao_acc_list
    df['iter_' + str(iteration+1) + '_ao_median_accuracy'] = ao_med_list
    df['iter_' + str(iteration+1) + '_ao_time'] = ao_time_list
    df['iter_' + str(iteration+1) + '_torch_avg_accuracy'] = torch_acc_list
    df['iter_' + str(iteration+1) + '_torch_median_accuracy'] = torch_med_list
    df['iter_' + str(iteration+1) + '_torch_time'] = torch_time_list
    df['iter_' + str(iteration+1) + '_colab_accuracy'] = colab_acc_list
    df['iter_' + str(iteration+1) + '_colab_time'] = colab_time_list
    print(df.head(5))

df.to_csv('smaller_per_user.csv', index=False)
