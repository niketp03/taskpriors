import torch
from torch import nn
from torch.utils.data import TensorDataset

from taskpriors import analyze

# Dummy model & dataset for illustration
model = nn.Linear(10, 2)
data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

# Create random labels
random_labels = torch.randint(0, 2, (100,))
# Create dataset with random features and labels
data = TensorDataset(torch.randn(100, 10), random_labels)

dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

stats = analyze(model, dataloader) 
print(stats)
