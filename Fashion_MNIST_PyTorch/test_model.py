#Tests the model

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

test_data = datasets.FashionMNIST(
    root = 'dataset/',
    train = False, 
    download = True,
    transform = ToTensor(),
    )

class ConvNN(nn.Module):

    def __init__(self, in_channels = 1, num_classes = 10):
        super(ConvNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 7, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),

            nn.Linear(256*3*3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.conv_stack(x)

model = ConvNN().to(device)

model.load_state_dict(torch.load("model_parameters/model_CNN.pth"))
model.eval()
x, y = test_data[45][0], test_data[45][1]
x = x.to(device)    #model and data should be on the same device

with torch.no_grad():
    pred = model(x[None, ...])
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')