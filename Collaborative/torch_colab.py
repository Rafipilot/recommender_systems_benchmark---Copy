import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection, preprocessing, metrics
import torch.nn as nn
import time
from data_prep import prepare_data



# Dataset class
class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id'].values, dtype=torch.long)
        self.genres = torch.tensor(np.stack(df['genres_enc'].values), dtype=torch.float32)
        self.lang = torch.tensor(np.stack(df['lang_enc'].values), dtype=torch.float32)
        self.vote_count = torch.tensor(np.stack(df['vote_count_enc'].values), dtype=torch.float32)
        self.vote_avg = torch.tensor(np.stack(df['vote_avg_enc'].values), dtype=torch.float32)
        self.rating = torch.tensor(np.stack(df['rating'].values), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'movie': self.movies[idx],
            'genres': self.genres[idx],
            'lang': self.lang[idx],
            'vote_count': self.vote_count[idx],
            'vote_avg': self.vote_avg[idx],
            'rating': self.rating[idx]
        }


# Model architecture
class RecSysModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 32)
        self.movie_emb = nn.Embedding(num_movies, 32)

        # Feature dimensions: genres(10) + lang(3) + vote_count(10) + vote_avg(3) = 27
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 26, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Sigmoid()
        )

    def forward(self, user, movie, genres, lang, vote_count, vote_avg):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        features = torch.cat([u, m, genres, lang, vote_count, vote_avg], dim=1)
        return self.fc(features)

def run_colab_model(num_users, reviews_per_user):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = prepare_data(num_user=num_users, reviews_per_user=reviews_per_user, per_user=False)

    # Label encoding
    user_encoder = preprocessing.LabelEncoder()
    movie_encoder = preprocessing.LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['userId'])
    df['movie_id'] = movie_encoder.fit_transform(df['movieId'])

    # Train/Test split
    train_df, val_df = model_selection.train_test_split(df, test_size=0.2, random_state=42)
    train_ds = MovieDataset(train_df)
    val_ds = MovieDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    # Initialize model
    model = RecSysModel(
        num_users=len(user_encoder.classes_),
        num_movies=len(movie_encoder.classes_)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    epochs = 25
    epoch_start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            inputs = {k: v.to(device) for k, v in batch.items() if k != 'rating'}
            output = model(**inputs)
            loss = criterion(output.squeeze(), batch['rating'].to(device))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluation
    model.eval()
    val_preds = []
    val_targets = []
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'rating'}
            outputs = model(**inputs).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(batch['rating'].cpu().numpy())

    val_preds = np.array(val_preds)
    pred_sum = np.sum(val_preds >= 0.5, axis=1)
    val_targets = np.array(val_targets)
    val_sum = np.sum(val_targets, axis=1)
    print(f'pred : {pred_sum}')
    print(f' target : {val_sum}')
    diff = np.abs(pred_sum - val_sum)
    count += np.sum(diff <= 2)
    time_taken = time.time() - epoch_start
    print(f'val_targets.shape : {len(val_targets)}, val_preds.shape : {len(val_preds)}')
    accuracy = count/len(val_targets)

    return accuracy, time_taken

if __name__=="__main__":
    accuracies = {}
    times = {}
    num_user_list = [250]
    num_reviews_list = [5, 10, 25, 100, 200]
    for i in num_user_list:
        for j in num_reviews_list:
            acc, t = run_colab_model(i, j)
            print(f'accuracy for {i} num users and {j} reviews per user is {acc}')
            print(f'time taken was {t}')
            accuracies[str(i)+" num_users + "+str(j)+" reviews per user"] = acc
            times[str(i) + " num_users + " + str(j) + " reviews per user"] = t

    print(accuracies)
    print(times)