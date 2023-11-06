import os
import librosa
import pandas as pd
import numpy as np
import torch
from torch import nn
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
print(torch.cuda.get_device_name(0))

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

# Create character set
characters = list(set(char for label in y_train for char in label)) 
characters.append('<PAD>')

# Create label encoding dicts
char_to_id = {char: id for id, char in enumerate(characters)}
id_to_char = {id: char for char, id in char_to_id.items()}

# Encode labels as integers
y_train_ids = [[char_to_id[char] for char in label] for label in y_train] 

# Pad sequences
max_len = max(max(len(mfcc) for mfcc in x_train), max(len(label) for label in y_train_ids))
y_train_padded_ids = pad_sequences(y_train_ids, maxlen=max_len, padding='post', value=char_to_id['<PAD>'])

# Reshape to 2D array with 1 column 
y_train_padded_ids = y_train_padded_ids.reshape(-1, 1) 

# One hot encode
onehot_encoder = OneHotEncoder()
# Fit on 2D array of IDs
onehot_encoder.fit(np.array(list(id_to_char.keys())).reshape(-1, 1))  
# Transform padded IDs to one-hot encoding
y_train_onehot = onehot_encoder.transform(y_train_padded_ids).toarray()

y_train_onehot_padded = pad_sequences(y_train_onehot, maxlen=max_len, padding='post')
# Pad input features
x_train_padded = pad_sequences(x_train, maxlen=max_len, padding='post')

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Use only the last output of each sequence for classification
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
model = Model(13, 128, len(characters)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Convert your data to PyTorch tensors and move them to the device
x_train_tensor = torch.tensor(x_train_padded).to(device)
y_train_tensor = torch.tensor(y_train_onehot_padded).to(device)

# Define the OneHotEncoder object
encoder = OneHotEncoder()

# Fit the encoder on the label data
encoder.fit(y_train_padded_ids)

# Correctly one-hot encode y_train_padded_ids
y_train_onehot = [encoder.transform(label.reshape(-1, 1)).toarray() for label in y_train_padded_ids]
y_train_onehot_padded = np.stack(y_train_onehot)

# Define a new, smaller batch size
new_batch_size = 10000

# Take a subset of your training data
y_train_onehot_padded_subset = y_train_onehot_padded[:new_batch_size]

print("New batch size: ", new_batch_size)
# Check the shape of y_train_onehot_padded_subset before it's tiled
print(f"Shape of y_train_onehot_padded_subset: {y_train_onehot_padded_subset.shape}")

for epoch in range(10000):
    outputs = model(x_train_tensor.float())  # Compute outputs in each iteration
    outputs = outputs.float()  # Convert to float


    y_train_tensor = y_train_tensor.view(-1).long()  # Convert to long

    # If the shapes do not match, reshape either outputs or y_train_tensor
    if outputs.shape[0] != y_train_tensor.shape[0]:
        # Reshape y_train_tensor to match the shape of outputs
        y_train_tensor = y_train_tensor[:outputs.shape[0]]

    # Now you can calculate your loss
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Accuracy: {accuracy_score(y_train_tensor.cpu().numpy(), outputs.argmax(1).cpu().numpy()) * 100:.2f}%")

torch.save(model.state_dict(), "model.pth")