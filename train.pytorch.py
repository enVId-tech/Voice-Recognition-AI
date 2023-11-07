import os
import librosa
import pandas as pd
import numpy as np
import torch
from torch import nn
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
print(torch.cuda.get_device_name(0))

audio_dir = "./AUDIO"
csv_file = "./TEXT/AUDIO.csv"

audio_files = os.listdir(audio_dir)

x_train = []
y_train = []

df = pd.read_csv(csv_file)

for file in audio_files:
    if not file.endswith(".mp3"):
        continue

    file_path = os.path.join(audio_dir, file)

    y, sr = librosa.load(file_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.transpose(mfcc, (1, 0))

    x_train.append(torch.tensor(mfcc))

    matched_text = df.loc[df['Video Matching'] == file, 'Text'].values[0]

    y_train.append(matched_text)

characters = list(set(char for label in y_train for char in label))
characters.append('<PAD>')

char_to_id = {char: id for id, char in enumerate(characters)}
id_to_char = {id: char for char, id in char_to_id.items()}

y_train_ids = [[char_to_id[char] for char in label] for label in y_train]
max_len = max(max(len(mfcc) for mfcc in x_train), max(len(label) for label in y_train_ids))

y_train_padded_ids = pad_sequences(y_train_ids, maxlen=max_len, padding='post', value=char_to_id['<PAD>'])
y_train_padded_ids = y_train_padded_ids.reshape(-1, 1)

onehot_encoder = OneHotEncoder()
onehot_encoder.fit(np.array(list(id_to_char.keys())).reshape(-1, 1))

y_train_onehot = onehot_encoder.transform(y_train_padded_ids).toarray()
y_train_onehot_padded = pad_sequences(y_train_onehot, maxlen=max_len, padding='post')

x_train_padded = pad_sequences(x_train, maxlen=max_len, padding='post')

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = Model(13, 64, len(characters), num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))

x_train_tensor = torch.tensor(x_train_padded).to(device)
y_train_tensor = torch.tensor(y_train_onehot_padded).to(device)

encoder = OneHotEncoder()
encoder.fit(y_train_padded_ids)

y_train_onehot = [encoder.transform(label.reshape(-1, 1)).toarray() for label in y_train_padded_ids]
y_train_onehot_padded = np.stack(y_train_onehot)
new_batch_size = 10000
y_train_onehot_padded_subset = y_train_onehot_padded[:new_batch_size]

epoch_amount = 100

for epoch in range(epoch_amount):
    outputs = model(x_train_tensor.float())
    outputs = outputs.float()

    y_train_tensor = y_train_tensor.view(-1).long()

    if outputs.shape[0] != y_train_tensor.shape[0]:
        y_train_tensor = y_train_tensor[:outputs.shape[0]]

    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs = outputs.argmax(1)

    # Get 20 of the audio files to test and check the accuracy
    test_files = audio_files[:40]

    x_test = []
    y_test = []

    for file in test_files:
        if not file.endswith(".mp3"):
            continue

        file_path = os.path.join(audio_dir, file)

        y, sr = librosa.load(file_path, sr=None, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.transpose(mfcc, (1, 0))

        x_test.append(torch.tensor(mfcc))

        matched_text = df.loc[df['Video Matching'] == file, 'Text'].values[0]

        y_test.append(matched_text)

    y_test_ids = [[char_to_id[char] for char in label] for label in y_test]
    y_test_padded_ids = pad_sequences(y_test_ids, maxlen=max_len, padding='post', value=char_to_id['<PAD>'])
    y_test_padded_ids = y_test_padded_ids.reshape(-1, 1)

    y_test_onehot = onehot_encoder.transform(y_test_padded_ids).toarray()
    y_test_onehot_padded = pad_sequences(y_test_onehot, maxlen=max_len, padding='post')

    x_test_padded = pad_sequences(x_test, maxlen=max_len, padding='post')

    x_test_tensor = torch.tensor(x_test_padded).to(device)
    y_test_tensor = torch.tensor(y_test_onehot_padded).to(device)

    with torch.no_grad():
        outputs = model(x_test_tensor.float())
        outputs = outputs.float()

        y_test_tensor = y_test_tensor.view(-1).long()

        if outputs.shape[0] != y_test_tensor.shape[0]:
            y_test_tensor = y_test_tensor[:outputs.shape[0]]

        accuracy = (outputs.argmax(1) == y_test_tensor).sum().item() / y_test_tensor.shape[0]
        print(f"Epoch: {epoch + 1}/{epoch_amount}, Loss: {loss.item()}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "model.pth")