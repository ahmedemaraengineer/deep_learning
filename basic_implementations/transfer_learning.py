import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision


# We want to cancel out the avg-pooling layer
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load the pretrained model
model = torchvision.models.vgg16(pretrained=True)

# Freezing all of the model's parameters
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
# Adding three extra layers that are compatible with our classification task
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))

# setting the device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# if load_model:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"))

for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
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
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got{num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
