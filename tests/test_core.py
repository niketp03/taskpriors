import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from taskpriors.core import (
    compute_kernel_matrix,
    compute_expectation_variance,
    analyze,
)


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("kernel_type", ["linear", "cosine", "rbf"])
def test_compute_kernel_matrix_identity(kernel_type):
    x = torch.tensor([[1.0], [2.0], [3.0]])
    ds = TensorDataset(x, torch.randint(0, 2, (3,)))
    loader = DataLoader(ds, batch_size=2)
    model = IdentityModel()
    K = compute_kernel_matrix(model, loader, kernel=kernel_type)
    feats = x
    if kernel_type == "linear":
        expected = feats @ feats.t()
    elif kernel_type == "cosine":
        feats_norm = feats / feats.norm(dim=1, keepdim=True)
        expected = feats_norm @ feats_norm.t()
    else:
        dists = torch.cdist(feats, feats, p=2)
        sigma = torch.median(dists)
        expected = torch.exp(-dists / (2 * sigma * sigma))
    assert torch.allclose(K, expected, atol=1e-6)


def test_compute_kernel_matrix_invalid_kernel():
    x = torch.randn(2, 2)
    ds = TensorDataset(x, torch.tensor([0, 1]))
    loader = DataLoader(ds, batch_size=1)
    model = IdentityModel()
    with pytest.raises(ValueError):
        compute_kernel_matrix(model, loader, kernel="unsupported")


def test_compute_expectation_variance():
    K = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])
    M = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    T = 1.0
    sigma = torch.sigmoid(K / T)
    exp_expected = (M * sigma).sum()
    var_expected = ((M ** 2) * sigma * (1 - sigma)).sum()
    exp, var = compute_expectation_variance(K, M, T)
    assert torch.allclose(exp, exp_expected)
    assert torch.allclose(var, var_expected)


def test_analyze_with_prior_kernel():
    x = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([0, 1])
    ds = TensorDataset(x, labels)
    loader = DataLoader(ds, batch_size=2)
    model = IdentityModel()
    prior = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = analyze(model, loader, prior_kernel=prior, T=1.0, kernel="linear")
    feats = x
    M = feats @ feats.t()
    sigma = torch.sigmoid(M / 1.0)
    exp_expected = (prior * sigma).sum().item()
    var_expected = ((prior ** 2) * sigma * (1 - sigma)).sum().item()
    assert pytest.approx(exp_expected) == result["expectation"]
    assert pytest.approx(var_expected) == result["variance"]


def test_analyze_binary_kernel():
    x = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    ds = TensorDataset(x, labels)
    loader = DataLoader(ds, batch_size=4)
    model = IdentityModel()
    result = analyze(model, loader, prior_kernel=None, T=1.0, kernel="linear")
    K = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    feats = x
    M = feats @ feats.t()
    sigma = torch.sigmoid(M / 1.0)
    exp_expected = (K * sigma).sum().item()
    var_expected = ((K ** 2) * sigma * (1 - sigma)).sum().item()
    assert pytest.approx(exp_expected) == result["expectation"]
    assert pytest.approx(var_expected) == result["variance"]
