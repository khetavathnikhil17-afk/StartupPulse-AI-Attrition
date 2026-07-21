# Contributing to StartupPulse AI

Thank you for your interest in contributing to StartupPulse AI. This guide covers the standards and workflow for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/StartupPulse-AI-Attrition.git
   cd StartupPulse-AI-Attrition
   ```
3. Create a branch for your change:
   ```bash
   git checkout -b feature/<descriptive-name>
   ```

## Development Environment

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- 4 GB+ RAM recommended (for TensorFlow)

### Setup

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Preprocess data
python -m src.data.preprocessing

# Train the model
python -m src.model.train

# Evaluate the model
python -m src.model.evaluate

# Launch the dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
StartupPulse-AI-Attrition/
├── src/                    # Source code modules
│   ├── data/               # Data loading and preprocessing
│   ├── model/              # DNN training, evaluation, prediction
│   ├── visualization/      # EDA and plot generation
│   ├── explainability/     # SHAP integration
│   └── utils/              # Configuration and logging
├── dashboard/              # Streamlit dashboard application
│   ├── app.py              # Main entry point
│   ├── _pages/             # Individual dashboard pages
│   └── components/         # Reusable widgets and CSS
├── data/                   # Raw and processed datasets
├── models/                 # Trained model artifacts
├── reports/                # Generated figures and metrics
└── requirements.txt        # Python dependencies
```

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints on all function signatures:
  ```python
  def load_data(path: Path) -> pd.DataFrame:
  ```
- Write docstrings for all public functions following the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):
  ```python
  def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
      """Compute classification metrics for binary predictions.

      Args:
          y_true: Ground truth binary labels.
          y_pred: Predicted binary labels.

      Returns:
          Dictionary containing accuracy, precision, recall, and F1.
  """
  ```
- Use `logging` instead of `print()` for runtime output
- Keep imports organized: stdlib, third-party, local (separated by blank lines)
- Use `Path` from `pathlib` for all file paths (no string concatenation)

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Modules | `snake_case` | `shap_explainer.py` |
| Functions | `snake_case` | `compute_class_weights()` |
| Classes | `PascalCase` | `AttritionPredictor` |
| Constants | `UPPER_SNAKE_CASE` | `MODEL_DIR` |
| Private | Leading underscore | `_save_fig()` |

### Dashboard

- Keep page logic in `dashboard/_pages/`
- Reuse components from `dashboard/components/`
- Use `st.cache_data` or `st.cache_resource` for expensive computations
- Follow existing glassmorphism CSS patterns

## Making Changes

1. Write clean, focused commits with clear messages:
   ```bash
   git commit -m "Add: batch prediction CLI support"
   ```

2. Ensure your changes work with the existing pipeline:
   ```bash
   python -m src.data.preprocessing
   python -m src.model.train
   streamlit run dashboard/app.py
   ```

3. Keep commits atomic — one logical change per commit

4. Do not commit model artifacts (`.keras`, `.pkl`), raw data, or virtual environment files

## Pull Request Process

1. Update your fork with the latest upstream changes:
   ```bash
   git remote add upstream https://github.com/<owner>/StartupPulse-AI-Attrition.git
   git fetch upstream
   git merge upstream/main
   ```

2. Push your branch to your fork:
   ```bash
   git push origin feature/<descriptive-name>
   ```

3. Open a Pull Request on GitHub with:
   - A descriptive title (e.g., "Add: XGBoost model comparison module")
   - A summary of what changed and why
   - Screenshots if UI changes are involved
   - Reference any related issues (e.g., "Closes #12")

4. Address review feedback promptly. Maintainers may request changes before merging.

5. PRs require review approval before merge. Do not self-merge.

## Issue Guidelines

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Full error traceback
- Relevant file paths and line numbers

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Suggested implementation approach (if any)

### Labels

| Label | Description |
|---|---|
| `bug` | Something is broken |
| `enhancement` | New feature or improvement |
| `documentation` | Docs update needed |
| `question` | Further information requested |
| `good first issue` | Suitable for new contributors |

---

Thank you for contributing to StartupPulse AI.
