import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from custom_dataset import CatsAndDogsDataset

my_transforms = transforms.Compose([
    # PIL Transformation to apply the transformations
    transforms.ToPILImage(),
    # Adjusting the resolution
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    # Randomly adjusting the brightness, contrast, saturation ..
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    # Horizontally flipping with a specific probability
    transforms.RandomHorizontalFlip(p=0.5),
    # Vertically flipping with a specific probability
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    # Transforming into a pytorch tensor
    transforms.ToTensor(),
    # It normalizes the images 3-times for each channel with these corresponding three values of mean and std
    # ( img[channel] - mean[channel] / std[channel] )
    transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])
])

dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=my_transforms)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img' + str(img_num) + '.png')
        img_num += 1


