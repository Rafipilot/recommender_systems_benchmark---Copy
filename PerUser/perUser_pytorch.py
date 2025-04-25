from statistics import median
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data_prep import prepare_data

# Check if GPU is available and set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# if device.type == "cuda":
#     print("GPU Name:", torch.cuda.get_device_name(0))


def run_torch_per_user(num_users, reviews_per_user, split=0.8):
    Users_data = prepare_data(num_user=num_users, reviews_per_user=reviews_per_user, per_user=True)

    # Define the PyTorch model (similar architecture to the Keras model)
    class MovieModel(nn.Module):
        def __init__(self):
            super(MovieModel, self).__init__()
            self.fc1 = nn.Linear(26, 32)  # 18 = 10 (genre) + 3 (vote_avg) + 3 (lang) + 2 (vote_count)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 10)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    correct_array = []  # Accuracy for each user group

    # Process each user group
    now = time.time()
    for idx, user_data in enumerate(Users_data):
        # print(f"\nUser Group {idx}:")
        n = len(user_data)
        split_index = math.floor(n * split)
        train_data = user_data[:split_index]
        test_data = user_data[split_index:]

        train_inputs = []
        train_labels = []

        for row in train_data:
            # Build input features (concatenate the encodings)
            genre_encoding = row[3]  # length 10
            vote_avg_encoding = row[5]
            lang_encoding = row[4]
            vote_count_encoding =row[6]
            rating_encoding = row[2]
            input_vector = np.concatenate((genre_encoding, vote_avg_encoding,lang_encoding,vote_count_encoding))
            if rating_encoding == 1:
                rating_encoding = 10 * [1]
            else:
                rating_encoding = 10 * [0]

            train_inputs.append(input_vector)
            train_labels.append(rating_encoding)  # 10-element target

        # Convert training data to torch tensors and send to device
        X_train = torch.Tensor(np.array(train_inputs))
        y_train = torch.Tensor(np.array(train_labels))

        # Create model, loss function and optimizer
        model = MovieModel()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop for 5 epochs
        model.train()
        for epoch in range(25):
            optimizer.zero_grad()
            try:
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch + 1}/25, Loss: {loss.item():.4f}")
            except Exception as e:
                print(e)

        # Testing phase for the current user group
        model.eval()
        correct = 0
        with torch.no_grad():
            for row in test_data:

                genre_encoding = row[3]
                vote_avg_encoding = row[5]
                lang_encoding =row[4]
                vote_count_encoding = row[6]
                input_vector = np.concatenate((genre_encoding, vote_avg_encoding,lang_encoding,vote_count_encoding))

                # Prepare input tensor (shape: [1, 18])
                X_test = torch.Tensor(np.array([input_vector]).reshape(1, -1))
                pred = model(X_test).cpu().numpy()[0]
                pred_sum = np.sum(pred >= 0.5)  # count number of neurons above threshold 0.5
                predicted_label = 1 if pred_sum >= 5 else 0

                # Ground truth: based on the rating encoding threshold
                rating_encoding = row[2]
                # if rating_encoding == 1:
                #     rating_encoding = 10 * [1]
                # else:
                #     rating_encoding = 10 * [0]
                # gt_sum = np.sum(rating_encoding)
                # true_label = 1 if gt_sum >= 5 else 0

                if predicted_label == rating_encoding:
                    correct += 1

        accuracy = correct / len(test_data) if test_data else 0
        correct_array.append(accuracy)
        # print(f"User group {idx} accuracy: {accuracy:.2f}")

    after = time.time()
    overall_accuracy = sum(correct_array) / len(correct_array) if correct_array else 0
    overall_median = median(correct_array)
    time_taken = after - now

    return overall_accuracy, overall_median, time_taken


if __name__ == "__main__":

    accuracies = {}
    times = {}
    num_user_list = [100, 200]
    num_reviews_list = [None]#, 500, 1000]
    for i in num_user_list:
        for j in num_reviews_list:
            try:
                acc, med, t = run_torch_per_user(i, j)
                print(f'accuracy for {i} num users and {j} reviews per user is {acc} and the median is {med}')
                print(f'time taken was {t}')
                accuracies[str(i) + " num_users + " + str(j) + " reviews per user"] = acc
                times[str(i) + " num_users + " + str(j) + " reviews per user"] = t
            except:
                pass

    print(accuracies)
    print(times)
