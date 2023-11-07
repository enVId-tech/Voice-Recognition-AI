import os
import librosa
import numpy as np
import torch
import pandas as pd
from torch import nn
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

# Define your Model class (similar to the one used during training)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # Number of LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The LSTM layers can be stacked with the `num_layers` parameter
        out, _ = self.lstm(x)
        # Use only the last output of each sequence for classification
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Initialize the model
# Define the characters that can appear in the audio files
characters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '_', '<pad>']

# Define the test data directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(13, 64, len(characters), num_layers=2).to(device)

audio_dir = "./AUDIO"
csv_file = "./TEXT/AUDIO.csv"

# Load the test data
audio_files = os.listdir(audio_dir)

# Load the CSV file
df = pd.read_csv(csv_file)

x_test = []
y_test = []

# Convert the audio to a tensor
def convert_audio_to_tensor(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs = np.transpose(mfccs, (1, 0))
    return torch.tensor(mfccs)

def convert_text_to_tensor(text):
    # Convert text to a list of IDs
    text_ids = []
    for char in text:
        if char.isalpha () or char.isspace (): # check if the character is a letter or a space
            text_ids.append (characters.index(char))
        elif char == "'": # keep the apostrophe
            text_ids.append (characters.index(char))
        else: # replace any other character with an underscore
            text_ids.append (characters.index('_'))
    # Convert to a tensor
    return torch.tensor(text_ids)

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

# Check the content of y_test
print(y_test[:5])

# Assuming words is a list of words in your vocabulary
words = sorted(set(word for sentence in y_test for word in sentence.split()))
word_to_index = {word: index for index, word in enumerate(words)}

# If y_test contains strings, convert them to integers
if all(isinstance(i, str) for i in y_test):
    y_test = [[word_to_index[word] for word in sentence.split()] for sentence in y_test]

# Try padding again
y_test_tensor = pad_sequences(y_test, padding='post', value=characters.index('<pad>'))

# Check the shapes of the tensors in x_test
print([x.shape for x in x_test[:5]])

# Pad the tensors in x_test to the same size
x_test_tensor = pad_sequence(x_test, batch_first=True)

# Check the device of the model
device = next(model.parameters()).device
print(f'Model is on device: {device}')

# Move the input tensor to the same device as the model
x_test_tensor = x_test_tensor.to(device)

# Pass the test data through the model
with torch.no_grad():
    outputs = model(x_test_tensor.float())

# Check if outputs is a list of tensors
if isinstance(outputs, list):
    # If outputs is a list of tensors, apply argmax(1) to each tensor in the list
    outputs = torch.stack([output.argmax(1) for output in outputs])
else:
    # If outputs is a single tensor, check its number of dimensions
    if outputs.dim() == 0:
        # If outputs is a scalar, add a dimension to it
        outputs = outputs.unsqueeze(0)
    elif outputs.dim() == 1:
        # If outputs only has one dimension, use argmax(0)
        outputs = outputs.argmax(0)
    elif outputs.dim() > 1:
        # If outputs has more than one dimension, use argmax(1)
        outputs = outputs.argmax(1)

# Calculate the accuracy of the model on the test data
total_correct = 0
total_elements = 0

for true_sequence, predicted_sequence in zip(y_test_tensor, outputs):
    true_sequence = true_sequence[true_sequence != characters.index('<pad>')]

    # Convert numpy arrays to PyTorch tensors
    if isinstance(true_sequence, np.ndarray):
        true_sequence = torch.from_numpy(true_sequence)
    if isinstance(predicted_sequence, np.ndarray):
        predicted_sequence = torch.from_numpy(predicted_sequence)

    # Check if both tensors have dimensions using ndim
    if true_sequence.ndim == 0:
        true_sequence = true_sequence.unsqueeze(0)
    if predicted_sequence.ndim == 0:
        predicted_sequence = predicted_sequence.unsqueeze(0)

    # Trim to match the length of true_sequence
    predicted_sequence = predicted_sequence[:len(true_sequence)]

    # Move tensors to the same device
    if true_sequence.device != predicted_sequence.device:
        true_sequence = true_sequence.to(predicted_sequence.device)

    # Check if both tensors are scalars before the comparison
    if true_sequence.dim() == 0 and predicted_sequence.dim() == 0:
        correct = torch.tensor(true_sequence == predicted_sequence).sum().item()
    else:
        correct = (true_sequence == predicted_sequence).sum().item()

    total_correct += correct
    total_elements += len(true_sequence)

# Check if total_elements is greater than zero before calculating accuracy
if total_elements > 0:
    accuracy = (total_correct / total_elements) * 100
    print(f"Test Accuracy: {accuracy:.10f}%")
else:
    print("No valid sequences to calculate accuracy.")