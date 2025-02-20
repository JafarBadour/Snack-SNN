import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from DST.layers.Snack import Snack
from DST.initializers import FixedDegreeRandomInitializer, UniformInitializer
from DST.pruner_grower import ZetaPrunerGrower
from benchmark.gpu.profiler import GPUAsyncProfiler


input_size = 28
output_size = 28


class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseLayer, self).__init__()

        # Initialize weights and bias
        self.weights = nn.Parameter(torch.randn(out_features, in_features))  # Random initialization
        self.bias = nn.Parameter(torch.zeros(out_features))  # Bias initialized to zeros

    def forward(self, x):
        """
        Manually calculate the forward pass: y = Wx + b
        x: Input tensor
        """
        # Perform the linear transformation: y = Wx + b
        output = torch.matmul(x, self.weights.T) + self.bias
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)


def train(type_: str, batch_size: int, hidden: int = 800, sparsity: float = 0, log=True, enable_dst=False):
    """

    :param batch_size:
    :param sparsity:
    :param type_: Dense ot Snack
    :return:
    """
    torch.manual_seed(0)
    # Define device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    FC1 = Snack(
        input_size * output_size,
        hidden,
        sparsity=sparsity,
        initializer=FixedDegreeRandomInitializer,
        device="cuda",
    )
    FC2 = Snack(
        hidden,
        10,
        sparsity=sparsity,
        initializer=FixedDegreeRandomInitializer,
        device="cuda",
    )

    # Define the model
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()

            if type_ == "Snack":

                self.fc1 = Snack(
                    input_size * output_size,
                    hidden,
                    sparsity=sparsity,
                    initializer=FixedDegreeRandomInitializer,
                    device="cuda",
                )
                self.fc2 = Snack(
                    hidden,
                    hidden,
                    sparsity=sparsity,
                    initializer=FixedDegreeRandomInitializer,
                    device="cuda",
                )

            elif type_ == "Dense":

                self.fc1 = DenseLayer(input_size * output_size, 128)
                self.fc1.weights.data = FC1.get_sp().dense().t().detach().clone()
                self.fc1.bias.data = FC1.bias.data.detach().clone()

                self.fc2 = DenseLayer(128, 10)
                self.fc2.weights.data = FC2.get_sp().dense().detach().clone().t()
                self.fc2.bias.data = FC2.bias.data.detach().clone()
            else:
                raise NotImplementedError("Not supported model type")
            self.fc3 = nn.Linear(hidden, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Flatten input

            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = NeuralNet().to(device)
    prof = GPUAsyncProfiler(0.2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # print(list(model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5

    for epoch in range(epochs):
        model.train()
        loss_item = None
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            loss_item = loss.item()

            optimizer.step()

            if batch_idx % 250 == 0:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                if log:
                    print(
                        f"""
                        =========================================
                        Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, 
                        Loss: {loss_item:.4f} trainable/total params {trainable_params}/{total_params}
                        =========================================
                        """
                    )
                if type_ == "Snack" and enable_dst:
                    # DST here
                    for sn in [model.fc1, model.fc2]:
                        pruner_grower = ZetaPrunerGrower(sn, zeta=0.1)
                        pruner_grower.prune()
                        pruner_grower.regrow(init=UniformInitializer, device="cuda")

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

    if log:
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    return correct / total


if __name__ == "__main__":

    from benchmark.gpu.profiler import GPUAsyncProfiler
    from tqdm import tqdm

    prof = GPUAsyncProfiler(0.2)
    # prof.start_benchmark()
    # train("Dense", 128)

    for sparsity in tqdm([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]):
        for enable_dst in [True, False]:
            acc = train("Snack", 128, sparsity=sparsity, log=False, enable_dst=enable_dst)
            print(f"Acc(Snack(sparsity={sparsity}, enable_dst={enable_dst})) = {acc}")
    # df = prof.stop_benchmark()
