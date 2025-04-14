print("Importing all libraries...")
import kagglehub
import torch
import pandas as pd
import numpy as np
import os
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection, preprocessing, metrics
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import time
from data_prep import prepare_data

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print("Imported!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for processing")

df = prepare_data(num_user=1000, reviews_per_user=500, per_user=False)

# Label encoding
user_encoder = preprocessing.LabelEncoder()
movie_encoder = preprocessing.LabelEncoder()
df['user_id'] = user_encoder.fit_transform(df['userId'])
df['movie_id'] = movie_encoder.fit_transform(df['movieId'])

# Dataset class
class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id'].values, dtype=torch.long)
        self.genres = torch.tensor(np.stack(df['genres_enc'].values), dtype=torch.float32)
        self.lang = torch.tensor(np.stack(df['lang_enc'].values), dtype=torch.float32)
        self.vote_count = torch.tensor(np.stack(df['vote_count_enc'].values), dtype=torch.float32)
        self.vote_avg = torch.tensor(np.stack(df['vote_avg_enc'].values), dtype=torch.float32)
        self.rating = torch.tensor(df['rating'].values, dtype=torch.float32)

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
print("Initialized data for training..")
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
            nn.Linear(32, 1)
        )

    def forward(self, user, movie, genres, lang, vote_count, vote_avg):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        features = torch.cat([u, m, genres, lang, vote_count, vote_avg], dim=1)
        return self.fc(features)


# Train/Test split
print("Train-Validation split...")
train_df, val_df = model_selection.train_test_split(
    df, test_size=0.2, stratify=df['rating'], random_state=42)

train_ds = MovieDataset(train_df)
val_ds = MovieDataset(val_df)

print("Loading data to model..")
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

# Initialize model
print("Initialized model...")
model = RecSysModel(
    num_users=len(user_encoder.classes_),
    num_movies=len(movie_encoder.classes_)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
epochs = 25
losses = []
print("Training started...")
epoch_start = time.time()
for epoch in range(epochs):
    start_time=time.time()
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

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time Taken: {time.time()-start_time:.2f} secs")
print(f"Took {time.time() - epoch_start:.2f} secs for {epochs} epochs")
# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.show()

# Evaluation
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'rating'}
        outputs = model(**inputs).squeeze().sigmoid()
        val_preds.extend(outputs.cpu().numpy())
        val_targets.extend(batch['rating'].cpu().numpy())

print(f"Validation ROC-AUC: {metrics.roc_auc_score(val_targets, val_preds):.4f}")
print(f"Validation Accuracy: {metrics.accuracy_score(val_targets, np.round(val_preds)):.4f}")

# Example prediction
test_sample = next(iter(val_loader))
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in test_sample.items() if k != 'rating'}
    outputs = model(**inputs).sigmoid()
    print("\nSample predictions:")
    print("Predicted:", outputs[:5].cpu().numpy().flatten())
    print("Actual:   ", test_sample['rating'][:5].numpy())