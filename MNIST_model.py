import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor 
from torch.utils.data import Dataloader
import matplotlib as plt



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
train_loader = Dataloader(
    train_data,
    batchesize=64,
    shuffle=True
)
test_loader = Dataloader(
    test_data,
    batchesize=64,
    shuffle=False
    
)

class BasicModel(nn.Module):
    def __init__(self):
        super.__init__()


plt.subplot()
