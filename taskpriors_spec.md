# Taskpriors Library Specification

The following markdown document fully specifies the initial scaffold for **taskpriors**, a pip‑installable Python library. Follow it verbatim to generate the project skeleton.

---

## 1. Project layout (src style)

```text
taskpriors/
├── src/taskpriors/__init__.py
├── src/taskpriors/core.py
├── tests/test_core.py
├── examples/basic_usage.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .github/workflows/ci.yml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

*All importable code lives under **``** to avoid “works‑on‑my‑machine” issues.*

---

## 2. Runtime dependencies

- `torch >= 2.0`
- `numpy >= 1.24`

These are the **only** entries in `requirements.txt` as well as the `[project.dependencies]` list inside `pyproject.toml`.

---

## 3. Packaging — `pyproject.toml`

- Use **Hatchling** as the build backend.
- Provide complete PEP 621 metadata.
- Add an `[project.optional-dependencies] dev` group containing:
  - `pytest`, `coverage[toml]`, `pytest-cov`, `ruff`, `black`
- Include a CLI entry‑point example (`yourcli = "taskpriors.cli:main"`), though the CLI file itself can be omitted for now.

---

## 4. Requirements files

```text
# requirements.txt – runtime only
torch>=2.0
numpy>=1.24
```

```text
# requirements-dev.txt – developer tools
-r requirements.txt
pytest
coverage[toml]
pytest-cov
ruff
black
build
twine
```

---

## 5. Testing & coverage

- \`\` folder with at least one placeholder unit test (`tests/test_core.py`).
- \`\` must contain:

```ini
[pytest]
addopts = --cov=taskpriors --cov-report=term-missing --cov-fail-under=80
```

CI and local runs must fail if total coverage < 80 %.

---

## 6. Core API stub

`src/taskpriors/core.py` defines:

```python
import torch
from torch.utils.data import Dataset

__all__ = ["analyze"]


def analyze(model: torch.nn.Module, dataset: Dataset) -> dict:
    """Analyze a task‑specific model/dataset pair and return statistics.

    TODO: implement the actual task‑prior analysis.
    """
    raise NotImplementedError("TODO: implement task analysis")
```

Export `analyze` in `__all__` and expose `__version__` in `src/taskpriors/__init__.py`.

---

## 7. Usage example

`examples/basic_usage.py` should demonstrate:

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset

from taskpriors import analyze

# Dummy model & dataset for illustration
model = nn.Linear(10, 2)
data = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))

stats = analyze(model, data)  # currently raises NotImplementedError
print(stats)
```

Keep the example runnable after installation with `pip install -e .`.

---

## 8. Continuous Integration (GitHub Actions)

`.github/workflows/ci.yml` must:

- Trigger on `push` and `pull_request` to `main`.
- Test on Python 3.9, 3.10, 3.11, 3.12.
- Steps:
  1. **Checkout**
  2. **Set up Python** for each matrix version.
  3. `pip install -r requirements.txt` then `pip install -r requirements-dev.txt` then `pip install -e .`
  4. Run `pytest` (coverage thresholds enforced by `pytest.ini`).
  5. Upload coverage to **Codecov** using `codecov/codecov-action@v4` (token in secret `CODECOV_TOKEN`).

---

## 9. README.md essentials

- Project name and tagline.
- Shields‑style badges:
  - CI status: `https://github.com/<user>/taskpriors/actions/workflows/ci.yml/badge.svg`
  - Codecov: `https://codecov.io/gh/<user>/taskpriors/branch/main/graph/badge.svg?token=XXXX`
- Install instructions:

```bash
pip install taskpriors
```

- Quick‑start snippet (the same as `examples/basic_usage.py`).
- Contribution guidelines placeholder linking to `requirements-dev.txt`.

---

## 10. License

Use the **MIT License**. Provide the standard text in `LICENSE`.

---

## 11. Optional: CHANGELOG

Maintain a `CHANGELOG.md` following *Keep a Changelog* format once the project evolves.

---

**End of specification – ready to hand to Codex or any scaffolding tool.**

