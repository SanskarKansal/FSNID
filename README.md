# FSNID

FSNID is a Python project for evaluating feature-selection methods on a synthetic network intrusion detection dataset. It compares multiple selectors (`fsnid`, `brown`, `firefly`, `lasso`, and `pi`) and then trains a classifier on the subset of retained features.

The repository also includes a compatibility fix for the Firefly selector when the requested number of fireflies exceeds the number of unique binary subsets.

## Overview

The project provides a single entry point, `main.py`, which:

- generates a synthetic intrusion-detection dataset when needed,
- runs a selected feature-selection method,
- splits the data into train/test sets,
- evaluates the selected features with a model,
- saves result files under the `data/` directory.

## Project structure

```text
FSNID/
├── data/                            # generated datasets and evaluation outputs
├── feature_selection_methods/       # feature selector implementations
│   ├── brown_fs.py
│   ├── firefly_fs.py
│   ├── fsnid_fs.py
│   ├── lasso_fs.py
│   └── pi_fs.py
├── evaluate_features.py             # model training and metric evaluation
├── generate_synthetic_data.py       # synthetic NID dataset generator
├── main.py                          # CLI entry point
├── mine.py                          # likely helper / exploratory pipeline
├── models.py                        # model definitions and wrappers
├── requirements.txt                 # Python dependencies
├── utils.py                         # utility functions
├── __init__.py
└── README.md
```

## Features

| Feature | Description |
| --- | --- |
| Multiple feature selectors | `fsnid`, `brown`, `firefly`, `lasso`, `pi` |
| Synthetic data generation | Creates `SYNTHETIC_X.npy` and `SYNTHETIC_Y.npy` when absent |
| Multiple classifier backends | `MLP`, `LSTM`, `TCN`, `GRU` |
| Parameterized training | Adjustable epochs, batch size, learning rate, and iterations |
| Result persistence | Saves metrics and selected feature indices for comparison |
| Firefly robustness fix | Handles combinatorial limits when `C(n, k)` is smaller than the requested firefly count |

## Requirements

This project is built for Python 3.9+ and depends on:

- numpy
- pandas
- scikit-learn
- torch

The exact dependency pins are listed in `requirements.txt`.

## Installation

```bash
git clone https://github.com/SanskarKansal/FSNID.git
cd FSNID

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

The main CLI is `main.py`.

```bash
python main.py \
  --selection_method fsnid \
  --model_type MLP \
  --num_iterations 10 \
  --epochs 15 \
  --learning_rate 0.001 \
  --batch_size 64
```

### Supported arguments

- `--nme`: dataset name prefix (default: `SYNTHETIC`)
- `--selection_method`: one of `fsnid`, `brown`, `firefly`, `lasso`, `pi`
- `--model_type`: one of `MLP`, `LSTM`, `TCN`, `GRU`
- `--num_iterations`: number of iterations for the selector
- `--epochs`: number of training epochs for the classifier
- `--learning_rate`: optimizer learning rate
- `--batch_size`: batch size for model training

### Example

```bash
python main.py --selection_method firefly --model_type MLP --epochs 10
```

If `data/SYNTHETIC_X.npy` and `data/SYNTHETIC_Y.npy` do not exist, the script automatically generates them by calling `generate_synthetic_nid_dataset()`.

## Dataset generation

`generate_synthetic_data.py` creates a synthetic intrusion-detection dataset with a mix of binary and continuous features. The generated data is saved as:

- `data/SYNTHETIC_X.npy`
- `data/SYNTHETIC_Y.npy`

The dataset generator creates a 10,000-sample classification problem with attack patterns embedded in informative binary/continuous feature combinations, plus noise and redundant features to make the task more realistic.

## Outputs

After each run, the project writes output files in `data/`:

- `<selection_method>_<nme>_metrics.json` — model metrics such as accuracy, precision, recall, F1, and false-positive rate
- `fsnid_<nme>_<selection_method>_time.npy` — feature-selection runtime in seconds
- `fsnid_fs_<nme>_<selection_method>.npy` — selected feature indices

For example, a default Firefly run produces files such as:

```text
data/firefly_SYNTHETIC_metrics.json
data/fsnid_SYNTHETIC_firefly_time.npy
data/fsnid_fs_SYNTHETIC_firefly.npy
```

## Notes on the Firefly fix

The Firefly implementation can fail when the number of requested fireflies is larger than the total number of unique binary subsets of size `k`.

The project includes a fallback that allows duplicates once the theoretical maximum number of unique subsets is exhausted, instead of raising an error. This preserves the algorithm's expected behavior when the search space is limited.

If you use this repository in a project or publication, please cite the repository and the underlying paper.

## Contributing

Contributions are welcome. If you are making changes, please:

1. fork the repository,
2. create a feature branch,
3. test your change with the relevant run command,
4. open a pull request with a clear summary.

## Project status

This repository is a research-oriented benchmarking and evaluation project for comparing feature-selection strategies on synthetic intrusion-detection data. It is primarily intended for experimentation and method evaluation rather than production deployment.