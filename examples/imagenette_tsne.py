"""
Example: t-SNE visualization of sampled labels on the Imagenette dataset.

This script loads the Imagenette data using torchvision's built-in dataset,
extracts features using a pretrained ResNet-18 backbone, samples labels
using the taskpriors sampler, and plots a 2D t-SNE embedding colored by the
sampled labels.

Dependencies:
    torchvision, scikit-learn, matplotlib

Usage:
    Run the script directly - it will automatically download Imagenette:
        python examples/imagenette_tsne.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from taskpriors.sampler import sample_labels_from_model

# Settings
batch_size = 64
temperature = 1.0
num_classes = 10
seed = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset using torchvision's built-in Imagenette
dataset = datasets.Imagenette(root="./data", split="train", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Build feature extractor from pretrained ResNet-18 (remove classification head)
backbone = models.resnet18(pretrained=True)
backbone.fc = torch.nn.Identity()  # Remove the final classification layer
model = backbone.to(device)

# Extract raw features for all samples
model.eval()
features = []
with torch.no_grad():
    for batch in dataloader:
        x = batch[0].to(device)
        feats = model(x).cpu().numpy()
        features.append(feats)
features = np.concatenate(features, axis=0)

# Sample labels via the Potts‑prior sampler
sampled_labels = sample_labels_from_model(
    model, dataloader,
    num_classes=num_classes,
    temperature=temperature,
    seed=seed,
    device=device,
)

# Compute 2D t-SNE embeddings of the raw features
tsne = TSNE(n_components=2, random_state=seed)
embeds = tsne.fit_transform(features)

# Plot
plt.figure(figsize=(8, 8))
for c in range(num_classes):
    idx = sampled_labels == c
    plt.scatter(embeds[idx, 0], embeds[idx, 1], label=str(c), s=5)
plt.legend(title="Sampled class")
plt.title("t-SNE of Imagenette features colored by sampled labels")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()