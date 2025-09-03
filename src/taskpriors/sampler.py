from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

__all__: Sequence[str] = (
    "center_kernel_factors",
    "sample_labels_from_kernel",
    "sample_labels_from_features",
    "sample_labels_from_model",
)

def center_kernel_factors(B: np.ndarray) -> np.ndarray:
    """Return *centered* low‑rank factors ``B_c``.

    Subtracts the column‑wise mean so that the induced kernel has zero row/column
    sums—a standard normalisation step for the Potts/task‑prior sampler.

    Parameters
    ----------
    B:
        ``(n, r)`` matrix whose rows are feature vectors.
    """

    if B.ndim != 2:
        raise ValueError("B must be a 2‑D array of shape (n, r)")
    return B - B.mean(axis=0, keepdims=True)

def sample_labels_from_kernel(
    B: np.ndarray,
    *,
    num_classes: int,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample one multi‑class labeling from *centerd* low‑rank factors.

    Implements the prefix sampler 

    ``P(y) ∝ exp(β · Tr(Yᵀ K Y))``  with  ``K ≈ B Bᵀ``  and  ``β = 1/T``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if num_classes < 2:
        raise ValueError("num_classes must be ≥ 2")

    B = np.ascontiguousarray(B, dtype=np.float64)
    n, r = B.shape
    beta = 1.0 / temperature

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    labels = np.empty(n, dtype=np.int64)
    U = np.zeros((r, num_classes), dtype=B.dtype)

    for i in order:
        h = beta * (B[i] @ U)  # (q,)
        h -= h.max()           # stabilise
        p = np.exp(h, dtype=B.dtype)
        p /= p.sum()

        c = rng.choice(num_classes, p=p)
        labels[i] = c
        U[:, c] += B[i]

    return labels

def sample_labels_from_features(
    features: Union[np.ndarray, torch.Tensor],
    *,
    num_classes: int,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    """Sample labels given a raw feature matrix.

    Parameters
    ----------
    features:
        ``(n, r)`` array‑like (NumPy or Torch).  Rows correspond to examples.
    center:
        If *True* (default) subtract the mean feature vector before sampling.
    num_classes, temperature, seed:
        Same semantics as :pyfunc:`sample_labels_from_kernel`.
    """

    if isinstance(features, torch.Tensor):
        feats_np = features.detach().cpu().numpy()
    else:
        feats_np = np.asarray(features)

    B = center_kernel_factors(feats_np) if center else np.asarray(feats_np)

    return sample_labels_from_kernel(
        B,
        num_classes=num_classes,
        temperature=temperature,
        seed=seed,
    )

def _identity(x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
    return x


def sample_labels_from_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    num_classes: int,
    temperature: float = 1.0,
    device: str | torch.device = "cuda",
    seed: Optional[int] = None,
    feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """Extract features with *model* then sample labels.

    Runs a single forward pass over *dataloader* in evaluation mode, collects
    their embeddings (optionally via *feature_fn*), centers them, and delegates
    to :pyfunc:`sample_labels_from_features`.
    """

    feature_fn = feature_fn or _identity
    was_training = model.training
    model.eval()

    dev = torch.device(device)
    model.to(dev)

    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, Sequence) else batch
            x = x.to(dev, non_blocking=True)
            feats.append(feature_fn(model(x)).cpu())

    if was_training:
        model.train()

    features = torch.cat(feats, dim=0)
    return sample_labels_from_features(
        features,
        num_classes=num_classes,
        temperature=temperature,
        seed=seed,
    )
