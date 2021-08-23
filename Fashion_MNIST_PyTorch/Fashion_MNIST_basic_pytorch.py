#Double hashed comments are not part of the project but rather extra attempts

#imports
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using the {device} device')

#Defining the hyperparameters
learning_rate = 0.001
epoch = 10
batch_size = 64

#train, test datasets and dataloaders
train_data = datasets.FashionMNIST(
    root='dataset/',
    train = True,
    download = True,
    transform = ToTensor(),
    )
test_data = datasets.FashionMNIST(
    root = 'dataset/',
    train = False, 
    download = True,
    transform = ToTensor(),
    )
train_dataloader = DataLoader(train_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

##Data Exploration
# for x, y in train_dataloader:
#     print(x.shape)
#     print(y.shape)
#     break

##create a Fully Connected Neural Network Model
# class NNModel(nn.Module):
#     def __init__(self):
#         super(NNModel, self).__init__()
#         self.flatten = nn.Flatten()
#         self.layer_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.ReLU()
#         )
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.layer_stack(x)
#         return logits
    
# model = NNModel().to(device)

##Create a Convolutional Neural Network 1(primitive)
# class ConvNN(nn.Module):

#     def __init__(self, in_channels = 1, num_classes = 10):
#         super(ConvNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
#         self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
#         self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
#         self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
#         self.fc2 = nn.Linear()
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = self.flatten(x)
#         x = self.fc1(x)

#         return x

#Create a Convolutional Neural Network 2
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
# print(model)

#optimize the model with losses and optimization function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#train the model
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
        y = y.to(device)
        
        #forward propagation
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad()   #In PyTorch, we need to set the gradients to zero before starting to do backpropragation 
                                #because PyTorch accumulates the gradients on subsequent backward passes.
        loss.backward()
        optimizer.step()    #gradient descent

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"\nloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

##Function to extensively test the model
#Alternatively made check_accuracy function which is a more general function to check
#the accuracy of both the training and test set

# def test(dataloader, model):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X = X.to(device)
#             y = y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#         test_loss /= size
#         correct /= size
#         print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def check_accuracy(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0
    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    model.train()

    correct /= size
    return correct

#Run the model
for e in range(epoch):
    print(f'Epoch {e+1}\n------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model)
print('Training Done!')

#Evaluating the model metrics
training_accuracy = check_accuracy(train_dataloader, model) * 100
test_accuracy = check_accuracy(test_dataloader, model) * 100
print(f'Accuracy on training set is = {training_accuracy:.2f}')
print(f'Accuracy on test set is = {test_accuracy:.2f}')

#save and load the model
torch.save(model.state_dict(), 'model_parameters/model_CNN.pth')
print('Saved the model to model_CNN.pth')

