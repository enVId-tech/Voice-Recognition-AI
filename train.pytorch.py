import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

# Set CUDA_LAUNCH_BLOCKING before importing torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  

# Set the environment variable to limit the maximum split size of the CachingAllocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Define a vocabulary for character-level tokenization
vocab = {'<pad>': 0, '<start>': 1, '<end>': 2}
for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ":
    vocab[char] = len(vocab)

# Hyperparameters
num_classes = len(vocab)  # Vocabulary size
batch_size = 2  # Reduced from 16 to 2
learning_rate = 3e-4
num_epochs = 100
max_seq_length = 300

class SpeechToTextModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechToTextModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))  # Adjust padding
        self.fc1 = None  # Will be defined later
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        
        # Define self.fc1 here, after we know the shape of x
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)  # Move to the same device as x
        
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class CustomSTTDataset(Dataset):
    def __init__(self, csv_path, audio_dir, max_seq_length):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.max_seq_length = max_seq_length

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx]['Video Matching'])
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        transcription = self.data.iloc[idx]['Text']

        # Tokenize the transcription to integers
        tokens = [vocab.get(char, vocab['<pad>']) for char in transcription]

        # Pad the sequence to the maximum length
        if waveform.size(1) < self.max_seq_length:
            padding = torch.zeros((waveform.size(0), self.max_seq_length - waveform.size(1)))
            waveform = torch.cat((waveform, padding), dim=1)

        return waveform, tokens

    def __len__(self):
        return len(self.data)
    
# Create the model and move it to the GPU (if available)
model = SpeechToTextModel(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Loss function and optimizer
criterion = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the dataset with a specified maximum sequence length
dataset = CustomSTTDataset(csv_path="TEXT/AUDIO.csv", audio_dir="AUDIO", max_seq_length=max_seq_length)

def collate_fn(batch):
    # Find the maximum length of any waveform in this batch
    max_length = max(waveform.size(1) for waveform, _ in batch)

    # Pad all waveforms and targets to the same length
    waveforms = []
    targets = []

    for waveform, tokens in batch:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
        input_length = waveform.size(1)
        target_length = len(tokens)

        # Pad inputs to the maximum length (max_length)
        if input_length < max_length:
            padding = torch.zeros((waveform.size(0), max_length - input_length), device=waveform.device)
            waveform = torch.cat((waveform, padding), dim=1)

        # Pad targets to the maximum length (max_length)
        if target_length < max_length:
            tokens += [vocab['<pad>']] * (max_length - target_length)

        waveforms.append(waveform)
        targets.append(tokens)  # Use append to keep the list of targets as a list of sequences

    return torch.stack(waveforms), targets

# Use DataLoader with the modified dataset and the new collate_fn
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)

        # Convert targets to tensor and flatten
        targets_flat = torch.tensor([t for tokens in targets for t in tokens]).to(device)

        # Calculate input_lengths and target_lengths for the CTCLoss
        input_lengths = torch.tensor([waveform.size(2) for waveform in inputs]).to(device)
        target_lengths = torch.LongTensor([len(targets[i]) for i in range(len(targets))]).to(device)

        print(f"input_lengths: {input_lengths}")
        print(f"target_lengths: {target_lengths}")


        # Pass the padded sequences through your model
        outputs = model(inputs.to(device))

        optimizer.zero_grad()
        loss = criterion(outputs, targets_flat, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")