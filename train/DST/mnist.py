import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from DST.layers.Snack import Snack
from DST.initializers.fixed_degree import FixedDegreeRandomInitializer


def train(type_: str, batch_size: int, sparsity: float = 0):
    """

    :param batch_size:
    :param sparsity:
    :param type_: Dense ot Snack
    :return:
    """
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Define the model
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            if type_ == "Snack":

                self.fc1 = Snack(
                    28 * 28,
                    128,
                    sparsity=sparsity,
                    initializer=FixedDegreeRandomInitializer,
                    device="cuda",
                )
                self.fc2 = Snack(
                    128,
                    10,
                    sparsity=sparsity,
                    initializer=FixedDegreeRandomInitializer,
                    device="cuda",
                )

            elif type_ == "Snack":

                self.fc1 = nn.Linear(28 * 28, 128)
                self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Flatten input
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = NeuralNet().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            print(data.shape, target.shape, device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train("Dense", 1)
