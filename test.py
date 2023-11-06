import os
import librosa
import numpy as np
import torch
import pandas as pd
from torch import nn
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Define your Model class (similar to the one used during training)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Initialize the model
# Define the characters that can appear in the audio files
characters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '_', '<pad>']

# Define the test data directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(13, 128, len(characters)).to(device)

audio_dir = "./AUDIO"
csv_file = "./TEXT/AUDIO.csv"

# Load the test data
audio_files = os.listdir(csv_file)

# Load the CSV file
df = pd.read_csv(csv_file)

# Convert the audio to a tensor
def convert_audio_to_tensor(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs = np.transpose(mfccs, (1, 0))
    return torch.tensor(mfccs)

# Convert the text to a tensor
def convert_text_to_tensor(text):
    # Convert text to a list of IDs
    text_ids = [characters.index(char) for char in text]
    # Convert to a tensor
    return torch.tensor(text_ids)

x_test = []
y_test = []

for file in audio_files:
    if not file.endswith(".mp3"):
        continue

    file_path = os.path.join(audio_dir, file)

    y, sr = librosa.load(file_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.transpose(mfcc, (1, 0))

    x_test.append(torch.tensor(mfcc))

    # Find the matching text in the CSV file
    matched_text = df.loc[df['Video Matching'] == file, 'Text'].values[0]

    y_test.append(matched_text)
    
# Pad the input features
x_test_tensor = pad_sequences(x_test, padding='post', value=characters.index('<pad>'))
# Pad the labels
y_test_tensor = pad_sequences(y_test, padding='post', value=characters.index('<pad>'))

# Pass the test data through the model
with torch.no_grad():
    outputs = model(x_test_tensor.float())

# Calculate the accuracy of the model on the test data
y_test_tensor = y_test_tensor.view(-1).long()  # Convert to long
accuracy = accuracy_score(y_test_tensor.cpu().numpy(), outputs.argmax(1).cpu().numpy()) * 100

print(f"Test Accuracy: {accuracy:.2f}%")
