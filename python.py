import os
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

# Make sure we are using GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.transpose(mfcc, (1, 0))

    x_train.append(mfcc)

    # Find the matching text in the CSV file
    matched_text = df.loc[df['Video Matching'] == file, 'Text'].values[0]

    y_train.append(matched_text)

from keras.utils import to_categorical
# Create a set of all unique characters in your text data
characters = set(char for label in y_train for char in label)


# Create a dictionary to map characters to unique IDs
char_to_id = {char: id for id, char in enumerate(characters)}

# Convert your labels to integer IDs
y_train_ids = [[char_to_id[char] for char in label] for label in y_train]

# Convert your labels to one-hot encoded format
y_train_onehot = [to_categorical(label, num_classes=len(characters)) for label in y_train_ids]

# Determine the length of the longest MFCC array
max_len = max(mfcc.shape[0] for mfcc in x_train)

# Pad the label sequences and convert to numpy array
y_train_padded = pad_sequences(y_train_onehot, maxlen=max_len, padding='post')

# Determine the length of the longest MFCC array
max_len = max(mfcc.shape[0] for mfcc in x_train)

# Pad the MFCC arrays and convert to numpy array
x_train = np.array([np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0))) for mfcc in x_train])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, 13), return_sequences=True)) 
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(characters), activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with tf.device('/device:GPU:0'):
    model.fit(x_train, y_train_padded, epochs=10000, batch_size=16)

model.save("model.h5")