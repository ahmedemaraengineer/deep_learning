import torch
import torchvision.transforms as transforms


a = torch.rand(3, 28, 28)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

print(transform(a).shape)
