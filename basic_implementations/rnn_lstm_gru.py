import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2
method = 'LSTM'
load_model = True


# Create simple CNN neural network:
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, method='RNN'):
        super(RNN, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if self.method == 'RNN':
            out, _ = self.rnn(x, h0)
        elif self.method == 'GRU':
            out, _ = self.gru(x, h0)
        else:
            out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def save_checkpoint(state, file_name="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, file_name)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# setting the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes, method).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"))

for epoch in range(num_epochs):
    losses = []
    # In case of interruption we want to save a checkpoint:
    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Updating using Adam algorithm
        optimizer.step()
    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:5f}')


# Checking accuracy on training & test to see how good our model is

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on the training data")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
