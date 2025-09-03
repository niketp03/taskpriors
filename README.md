# taskpriors

## Setup

### Prerequisites

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niketp03/taskpriors.git
   cd taskpriors
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   OR with uv
  ```bash
   uv venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   OR with uv
   ```bash
   uv sync --no-dev
   ```


### Usage

After installation, you can run the simple intro script:
    ```bash
    uv sync --no-dev
    ```

You can also now run any of the scripts used to generate the figures from the paper.
