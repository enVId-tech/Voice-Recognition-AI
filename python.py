import os
import librosa
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

audio_dir = "./AUDIO"
csv_file = "./TEXT/AUDIO.csv"

audio_files = os.listdir(audio_dir)

x_train = []
y_train = []

# Load the CSV file
df = pd.read_csv(csv_file)

for file in audio_files:
    if not file.endswith(".mp3"):
        continue

    file_path = os.path.join(audio_dir, file)

    y, sr = librosa.load(file_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.transpose(mfcc, (1, 0))

    x_train.append(torch.tensor(mfcc))

    # Find the matching text in the CSV file
    matched_text = df.loc[df['Video Matching'] == file, 'Text'].values[0]

    y_train.append(matched_text)

# Create a set of all unique characters in your text data
characters = list(set(char for label in y_train for char in label))

# Create a dictionary to map characters to unique IDs
char_to_id = {char: id for id, char in enumerate(characters)}

# Convert your labels to integer IDs
y_train_ids = [[char_to_id[char] for char in label] for label in y_train]

# Convert your labels to one-hot encoded format
encoder = OneHotEncoder(sparse=False)
y_train_onehot = [encoder.fit_transform(np.array(label).reshape(-1, 1)) for label in y_train_ids]

# Determine the length of the longest MFCC array
max_len = max(mfcc.shape[0] for mfcc in x_train)

# Pad the label sequences and convert to numpy array
y_train_padded = pad_sequence([torch.tensor(label) for label in y_train_onehot], batch_first=True, padding_value=0)

# Determine the length of the longest MFCC array
max_len = max(mfcc.shape[0] for mfcc in x_train)

# Pad the MFCC arrays and convert to numpy array
x_train = pad_sequence(x_train, batch_first=True, padding_value=0)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = Model(13, 128, len(characters)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

x_train = x_train.to(device)
y_train_padded = y_train_padded.to(device)

for epoch in range(10000):
    outputs = model(x_train)
    loss = criterion(outputs.view(-1, len(characters)), y_train_padded.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10000, loss.item()))

torch.save(model.state_dict(), "model.pth")
