import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from taskpriors.sampler import (
    centre_kernel_factors,
    sample_labels_from_kernel,
    sample_labels_from_model,
)


def test_centre_kernel_factors_basic():
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Bc = centre_kernel_factors(B)
    expected = B - B.mean(axis=0, keepdims=True)
    assert np.allclose(Bc, expected)
    # Check column means are zero
    assert np.allclose(Bc.mean(axis=0), 0.0)
    # Shape is preserved
    assert Bc.shape == B.shape


@pytest.mark.parametrize("arr", [np.zeros(3), np.zeros((2, 2, 2))])
def test_centre_kernel_factors_invalid_dim(arr):
    with pytest.raises(ValueError):
        centre_kernel_factors(arr)


def test_sample_labels_from_kernel_invalid_params():
    B = np.zeros((3, 2))
    with pytest.raises(ValueError):
        sample_labels_from_kernel(B, num_classes=1)
    with pytest.raises(ValueError):
        sample_labels_from_kernel(B, num_classes=2, temperature=0)
    with pytest.raises(ValueError):
        sample_labels_from_kernel(B, num_classes=2, temperature=-1.0)


def test_sample_labels_from_kernel_reproducibility_and_shape():
    rng = np.random.RandomState(0)
    B = rng.standard_normal((10, 4))
    labels1 = sample_labels_from_kernel(B, num_classes=3, temperature=1.0, seed=42)
    labels2 = sample_labels_from_kernel(B, num_classes=3, temperature=1.0, seed=42)
    # Reproducible given same seed
    assert np.array_equal(labels1, labels2)
    # Correct shape and dtype
    assert isinstance(labels1, np.ndarray)
    assert labels1.shape == (10,)
    assert labels1.dtype == np.int64
    # Values within valid range
    assert labels1.min() >= 0 and labels1.max() < 3


def test_sample_labels_from_model_basic_pipeline():
    # Create simple data and identity model
    x = torch.randn(7, 5)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=3)
    model = nn.Identity()
    num_classes = 4
    temperature = 1.5
    seed = 123

    # Sample via model pipeline
    labels_model = sample_labels_from_model(
        model, loader, num_classes=num_classes, temperature=temperature, seed=seed, device="cpu"
    )

    # Direct sampling using centred kernel factors
    B = x.numpy()
    B = centre_kernel_factors(B)
    labels_direct = sample_labels_from_kernel(
        B, num_classes=num_classes, temperature=temperature, seed=seed
    )

    assert np.array_equal(labels_model, labels_direct)


def test_sample_labels_from_model_preserves_training_state():
    x = torch.randn(3, 2)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=1)
    model = nn.Linear(2, 2)
    # If model is training, state should be restored
    model.train()
    _ = sample_labels_from_model(model, loader, num_classes=2, temperature=1.0, seed=0, device="cpu")
    assert model.training
    # If model is eval, state should be preserved
    model.eval()
    _ = sample_labels_from_model(model, loader, num_classes=2, temperature=1.0, seed=0, device="cpu")
    assert not model.training


def test_sample_labels_from_model_with_feature_fn():
    x = torch.randn(6, 3)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=2)
    model = nn.Identity()
    def feature_fn(out):
        return out * 2.0

    num_classes = 3
    temperature = 2.0
    seed = 7

    labels_model = sample_labels_from_model(
        model, loader, num_classes=num_classes, temperature=temperature, seed=seed, device="cpu", feature_fn=feature_fn
    )

    B = (x * 2.0).numpy()
    B = centre_kernel_factors(B)
    labels_direct = sample_labels_from_kernel(
        B, num_classes=num_classes, temperature=temperature, seed=seed
    )

    assert np.array_equal(labels_model, labels_direct)


def test_sample_labels_from_model_invalid_params():
    x = torch.randn(2, 2)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=1)
    model = nn.Identity()
    with pytest.raises(ValueError):
        sample_labels_from_model(model, loader, num_classes=1, temperature=1.0, seed=0, device="cpu")
    with pytest.raises(ValueError):
        sample_labels_from_model(model, loader, num_classes=2, temperature=0.0, seed=0, device="cpu")