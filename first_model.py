import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# --- Data ---
train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data  = datasets.MNIST(root='data', train=False, download=False, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)


# --- Model ---
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


model = DigitNet()
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --- Training ---
def train(epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            preds = model(images)
            loss  = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")


# --- Evaluation ---
def evaluate():
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            preds   = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()

    accuracy = correct / len(test_data) * 100
    print(f"Test accuracy: {accuracy:.2f}%")


# --- Visualize ---
def visualize(n=10):
    model.eval()
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"pred:{preds[i]}\nreal:{labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


train(epochs=3)
evaluate()
visualize()
