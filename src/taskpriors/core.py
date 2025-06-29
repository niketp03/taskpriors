import torch
from torch.utils.data import Dataset
from typing import Tuple

__all__ = ["analyze"]

def analyze(model: torch.nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            prior_kernel: torch.Tensor = None,
            T: float = 1.0,
            kernel: str = "cosine") -> dict:
    """
    Analyze a task-specific model/dataset pair and return statistics.
            This function computes kernel-based statistics (expectation and variance) for a given model and dataset.
            It can work with either a provided prior kernel or generate a binary kernel matrix based on class labels.
            Args:
                model (torch.nn.Module): Neural network model to analyze
                dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset
                prior_kernel (torch.Tensor, optional): Pre-computed kernel matrix. Defaults to None.
                T (float, optional): Temperature parameter for kernel calculations. Defaults to 1.0.
                kernel (str, optional): Type of kernel to use ("cosine" etc.). Defaults to "cosine".
            Returns:
                dict: Dictionary containing:
                    - "expectation" (float): Expected value based on kernel calculations
                    - "variance" (float): Variance based on kernel calculations
            Notes:
                - If prior_kernel is None, function creates a binary kernel matrix based on class equality
                - Assumes dataloader returns tuples where second element contains labels
    """
    if prior_kernel is not None:
        M = compute_kernel_matrix(model, dataloader, kernel = kernel)
        expectation, variance = compute_expectation_variance(prior_kernel, M, T)
        return {
            "expectation": expectation.item(),
            "variance": variance.item()
        }
    else:
        M = compute_kernel_matrix(model, dataloader, kernel = kernel)
        # Collect labels
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                batch_labels = batch[1]  # Assumes second element contains labels
                labels.append(batch_labels)
                
        # Create binary kernel matrix based on class equality
        labels = torch.cat(labels, dim=0)
        K = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        expectation, variance = compute_expectation_variance(K, M, T)
        return {
            "expectation": expectation.item(),
            "variance": variance.item()
        }

def compute_kernel_matrix(model: torch.nn.Module, 
                          dataloader: torch.utils.data.DataLoader, 
                          kernel: str) -> torch.Tensor:
    """Compute the kernel matrix for the model and dataset.
    
    Args:
        model: PyTorch neural network model
        dataloader: DataLoader to compute kernel matrix for
        kernel: Type of kernel to use ('linear', 'rbf', 'cosine')
        
    Returns:
        torch.Tensor: Computed kernel matrix
    """
    if kernel not in ["linear", "rbf", "cosine"]:
        raise ValueError(f"Unsupported kernel type: {kernel}")
        
    # Set model to eval mode
    model.eval()
    features = []
    
    # Collect features
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]  # Assumes first element is input data, second is labels
            batch_features = model(inputs)
            features.append(batch_features)
    
    # Concatenate all features
    features = torch.cat(features, dim=0)
    
    # Compute kernel matrix
    if kernel == "linear":
        K = torch.mm(features, features.t())
    elif kernel == "rbf":
        # Compute RBF kernel
        dists = torch.cdist(features, features, p=2)
        sigma = torch.median(dists)
        K = torch.exp(-dists / (2 * sigma * sigma))
    else:  # cosine
        features_norm = features / features.norm(dim=1, keepdim=True)
        K = torch.mm(features_norm, features_norm.t())
    
    return K

def compute_expectation_variance(K: torch.Tensor,
                                 M: torch.Tensor,
                                 T: float = 1.0) -> Tuple[torch.Tensor,
                                                          torch.Tensor]:
    """Return Σ M σ and Σ M² σ(1‑σ) with σ = sigmoid(K/T)."""
    sigma       = torch.sigmoid(K / T)
    expectation = (M * sigma).sum()
    variance    = ((M**2) * sigma * (1 - sigma)).sum()
    return expectation, variance
    
