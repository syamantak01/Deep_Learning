#Double hashed comments are not part of the project but rather extra attempts

#imports
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Resize, Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


#set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using the {device} device')

#Defining the hyperparameters
learning_rate = 0.001
epoch = 10
batch_size = 64

transform = Compose(
    [Resize((224, 224)),    #the inputs to the VGG network are of the shape (224, 224), while that of CIFAR-10 dataset is (32, 32) and hence we resize them to (224, 224)
     ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])      #Normalizes the pixel values; We are using mean = 0.5 and sttandard deviation = 0.5 for all 3 channels

#train, test datasets and dataloaders
train_data = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



#load pretrained vgg16 model
model = models.vgg16(pretrained = True)

#freeze convolutional weights
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6].out_features = 10

model = model.to(device)
# print(model)
# sys.exit()



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



