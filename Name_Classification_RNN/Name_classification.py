import torch
from torch import nn
import matplotlib.pyplot as plt

from utils import characters, n_char
from utils import load_data, char_to_tensor, name_to_tensor, random_training_example

#set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

#Building RNN from scratch
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__() 

        self.hidden_size = hidden_size
        self.inp_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.inp_to_output = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim = 1)
    

    def forward(self, inp, hidden):
        combined = torch.cat((inp, hidden), 1)
        new_hidden = self.inp_to_hidden(combined)
        output = self.inp_to_output(combined)
        output = self.softmax(output)
        return output, new_hidden

names_by_language, languages = load_data()
n_languages = len(languages)

#Hyperparamters
n_hidden = 128
learning_rate = 0.005
n_iterations = 100000


model = RNN(n_char, n_hidden, n_languages).to(device)
# print(model)

#Optimize the model with loss and optimization function

#loss function used will be Negative Likely Neighbourhood
#Note: Cross Entropy Loss = Log(Softmax) + NLL Loss
#So instead of doing it in 2 steps we can also apply Cross Entropy Loss which takes care of both of these things
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train(name_tensor, language_tensor, model, loss_fn, optimizer):
    
    name_tensor = name_tensor.to(device)
    language_tensor = language_tensor.to(device)

    #initialise the hidden tensors as zeros
    hidden = torch.zeros(1, model.hidden_size)
    hidden = hidden.to(device)

    for i in range(name_tensor.shape[0]):
        output, hidden = model(name_tensor[i], hidden)
    
    loss = loss_fn(output, language_tensor)

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() #gradient descent

    return output, loss.item()

current_loss = 0
all_loses = []

for i in range(n_iterations): 
    language, name, language_tensor, name_tensor = random_training_example(names_by_language, languages)
    
    output, loss = train(name_tensor, language_tensor, model, loss_fn, optimizer)

    current_loss += loss

    if (i+1) % 1000 == 0:
        avg_loss = current_loss/1000
        all_loses.append(avg_loss)
        print(f"\nloss: {avg_loss:.4f}  iteration: [{i+1}/{n_iterations}]")
        current_loss = 0

#Visualise the loss
plt.figure()
plt.plot(all_loses)
plt.show()

#predicting the language wrt to a name
def predict(model, name):
    print(f'The name is: {name}')
    with torch.no_grad():
        name_tensor = name_to_tensor(name)
        name_tensor = name_tensor.to(device)

        hidden = torch.zeros(1, model.hidden_size)
        hidden = hidden.to(device)

        for i in range(name_tensor.size()[0]):
            output, hidden = model(name_tensor[i], hidden)

        pred = languages[torch.argmax(output).item()]

        return pred
    
#To test it
while True:
    name = input("Input: ")
    if name == "exit":
        break
    
    print(predict(model, name))
    

