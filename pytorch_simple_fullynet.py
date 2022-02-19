import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create simple fully-connected network:transforms
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_checkpoint(state, file_name="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, file_name)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# setting the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
load_model = True


# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"))

for epoch in range(num_epochs):
    losses = []
    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Flatten the images
        data = data.reshape(data.shape[0], -1)
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
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got{num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
