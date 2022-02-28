import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim


# LeNet architecture
# 1*32*32 Input -> (5*5), s = 1 ,p = 0 -> avg pool, s = 2, p = 0 ->
# (5*5), s = 1, p = 0 -> avg pool, s = 2, p = 0 -> Conv (1*1)
# (with 120 channels) * Linear 120 -> 84 * Linear 84 -> 10

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # num_examples * 120 * 1 * 1 -> num_examples * 120
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

batch_size = 64

train_dataset = datasets.MNIST(root='datset/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='datset/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

learning_rate = 0.001
epochs = 3

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    losses = []
    for idx, (data, target) in enumerate(train_loader):
        # forward
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:5f}')


# Checking accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'got {num_correct} from {num_samples} correct with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
