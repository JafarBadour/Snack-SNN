import torch
from DST.layers import Snack

def train():
    import random
    random.seed(42)
    input_size = 4
    output_size = 2
    model = Snack(input_size, output_size, sparsity=0.2).cuda()
    print(list(model.parameters()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Dummy data
    x = torch.randn(input_size).cuda()
    target = torch.randn(output_size).cuda()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            

if __name__ == "__main__":
    train()