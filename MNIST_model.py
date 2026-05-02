import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


# Just download a dataset from torchvision. Look at the documentation of torchvision to undesratnd params and why train and test datasets are separated
train_data = datasets.MNIST(
    train = True,
    transform = ToTensor(),
    root = 'data',
    download = True 
)
test_data = datasets.MNIST(
    train = False,
    transform = ToTensor(),
    root = 'data',
    download = True 
)

# A Wrap on our data to make it more easier for a model (compress a bit and make it able to shuffling)
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
    
)

sample, label = next(iter(train_loader))
print(sample.shape, label.shape)
# There is an architecture of our model based on nn Module from pytorch

class BasicModel(nn.Module):
    def __init__(self):
        # Inhering nn module functions and methods like "nn.Sequential"
        super().__init__()
        # It's just a layer defination
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()
    # There is a function which define how a data will be computed in layers and what do we get as an output
    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

model = BasicModel()

# resice gradince decsent from "öptimizer" func. Getting 2 params as scpecific of func we reciev from pytroch
# new_weight = old_weight - learning_rate * loss/weight
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # lr is learning rate
# Getting 2 params as specific of function
loss_func = nn.CrossEntropyLoss()


def train_model(epoches):
    for epoch in range(epoches):
        model.train()
        total_loss = 0
        # here we getting data as images from train_loader on wich we will train our model 
        for images, labels in train_loader:

            logits = model(images)
            loss = loss_func(logits,labels)

            # main training model loop:
            # 1. zero_grad() - clearing old grad from past step
            # 2. loss.backward() - figure out how wrong we are by retunring loss AND BACKWARD figure out which direction to move it to reduce that wrongness. (I might find out how a loss algorithm looks like)
            # 3. optimizer.step() - use gradience decsent to  nudge weights and biases
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # it's just a variable to monitor how training is going
            total_loss += loss.item()
        # it's a final variable for me to monitor which loss we eventualy got
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epoches}  loss: {avg_loss:.4f}")

def visualize():

    model.eval()
    # Where images is and labels is 
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    # shape of grapgh
    fig, axes = plt.subplots(1, 1, squeeze=False)

    n = random.randint(0, 63)
    axes[0][0].imshow(images[n].squeeze())
    axes[0][0].set_title(f"pred:{preds[n]}\nreal:{labels[n]}")
    axes[0][0].axis('off')

    plt.tight_layout()
    plt.show()

train_model(3) 
visualize()



    
