## PhenoGnet

arXiv: 2509.14037 ([https://arxiv.org/abs/2509.14037](https://arxiv.org/abs/2509.14037))
License: MIT ([https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))

This repository contains the official implementation for PhenoGnet, a novel graph-based contrastive learning framework designed to predict disease similarity.

PhenoGnet integrates gene functional interaction networks and the Human Phenotype Ontology (HPO) to learn powerful embeddings for genes and phenotypes. By aligning these two views, the model can compute disease similarity scores that capture complex biological relationships, outperforming existing state-of-the-art methods.
---

## Framework Overview

<p align="center">
  <img src="assets/PhenoGnet_overview.png" alt="PhenoGnet Framework Workflow" width="550">
</p>

*Figure: Overall workflow of the PhenoGnet framework integrating gene and phenotype graphs through contrastive learning.*

---
Model Architecture

PhenoGnet consists of two key components: an Intra-view Model to encode the gene and phenotype graphs separately, and a Cross-view Model to align them in a shared latent space.

1.  Intra-view Model:

      * Gene Network: A Graph Convolutional Network (GCN) is used to encode the gene functional interaction network (from HumanNet).
      * Phenotype Network: A Graph Attention Network (GAT) is used to encode the Human Phenotype Ontology (HPO) graph. The initial features for HPO terms are generated from their textual descriptions using Sentence-BERT (all-mpnet-base-v2).

2.  Cross-view Model:

      * A shared-weight multilayer perceptron (MLP) projects the embeddings from both the GCN and GAT into a common latent space.
      * Contrastive learning is applied to train the entire model. Known gene-phenotype associations are used as positive pairs, and randomly sampled unrelated pairs serve as negatives. This process "pulls" related gene and phenotype embeddings closer together and "pushes" unrelated ones apart.

3.  Disease Similarity Prediction:

      * Diseases are represented by the mean embedding (average-pooling) of their associated genes and/or phenotypes.
      * The similarity between any two diseases is calculated using the cosine similarity of their final embedding vectors.

## Installation

The project has been tested with Python 3.10. A virtual environment is recommended
so that the pinned scientific and PyTorch dependencies do not conflict with other
Python projects on your machine.

The requirements are split into a base file plus one PyTorch/PyG profile:

- `requirements.txt`: shared Python dependencies.
- `requirements-torch-cpu.txt` and `requirements-pyg-cpu.txt`: CPU-only install.
- `requirements-torch-cu121.txt` and `requirements-pyg-cu121.txt`: CUDA 12.1 install for NVIDIA GPUs.

Choose either the CPU profile or the CUDA profile for a given virtual environment.
If you switch profiles later, recreating `.venv` is usually the cleanest option.

1. Clone the repository:

    ```bash
    git clone git@github.com:masino-lab/PhenoGnet.git
    cd PhenoGnet
    ```

2. Create a virtual environment:

    ```bash
    python -m venv .venv
    ```

3. Activate the environment.

    On Windows PowerShell:

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

    If PowerShell blocks activation scripts, run this once for the current shell
    and then activate again:

    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1
    ```

    On Linux/macOS:

    ```bash
    source .venv/bin/activate
    ```

4. Upgrade pip and packaging tools:

    ```bash
    python -m pip install --upgrade pip setuptools wheel
    ```

5. Install the shared dependencies:

    ```bash
    python -m pip install -r requirements.txt
    ```

6. Install one PyTorch/PyG profile.

    For CPU-only installs on Windows or Linux:

    ```bash
    python -m pip install -r requirements-torch-cpu.txt
    python -m pip install -r requirements-pyg-cpu.txt
    ```

    For NVIDIA GPU installs with CUDA 12.1 wheels on Windows or Linux:

    ```bash
    python -m pip install -r requirements-torch-cu121.txt
    python -m pip install -r requirements-pyg-cu121.txt
    ```

    You do not need to install the CUDA Toolkit separately for these pip wheels,
    but your NVIDIA driver must be new enough for CUDA 12.1. Run `nvidia-smi` to
    confirm that your GPU and driver are visible. CUDA is not supported on macOS.

7. Verify the installation:

    ```bash
    python -m pip check
    python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('pyg', torch_geometric.__version__); print('scatter', torch_scatter.__version__); print('sparse', torch_sparse.__version__)"
    ```

    CPU installs should report `cuda available False`. CUDA installs should report
    `cuda available True`; if not, confirm that the active interpreter is the repo
    virtual environment (`.venv\Scripts\python.exe` on Windows or `.venv/bin/python`
    on Linux/macOS).

## Running

After activating the virtual environment and installing dependencies, run the main
training script from the repository root:

```bash
python Code/run.py
```

Running from the repository root is recommended because it keeps logs and other
relative-output files in one predictable place. The script now resolves its default
input and plot paths relative to the repository root, so this also works:

```bash
cd Code
python run.py
```

Before starting a long run, you can confirm the command-line options without
loading data or training:

```bash
python Code/run.py --help
```

A few common examples:

```bash
# Train with the default HNET encoder settings.
python Code/run.py --wandb_label baseline_hnet

# Force CPU even if CUDA is available.
python Code/run.py --disable-cuda

# Use the HPO encoder and a shorter test run.
python Code/run.py --encoder_mode hpo --epochs 5 --wandb_label hpo_smoke_test

# Run hyperparameter tuning instead of regular training.
python Code/run.py --hyperparameter_tuning --n_trials 30
```

By default, the script expects processed inputs under `data/processed/`, HPO text
embeddings under `data/hpo_embeddings/`, and writes evaluation plots to `plots/`.
Use `--data`, `--hpo_embeddings_path`, `--full_dataset`, and `--output_dir` to point
at alternate files or directories.

CUDA is selected automatically when the active environment has a CUDA-enabled
PyTorch build and `torch.cuda.is_available()` returns `True`. Add `--disable-cuda`
only when you intentionally want CPU training.

## Command-Line Arguments

`python Code/run.py --help` prints the short reference, including defaults. The
notes below explain when each option is useful.

- `--data`: Directory containing the processed graph and mapping files used for
  training, including `hnet.npz`, `hpo2hpo_rec.npz`, `g2hpo_all_ancestors.npz`,
  `dis2hpo.npz`, `dis2g.npz`, and the triples loaded by `load_triples`. Change
  this when using a different processed dataset or data location.
- `--h_dim`: Hidden embedding size for the HPO and gene encoders. Increase it for
  more model capacity; decrease it for quicker or lower-memory experiments.
- `--z_dim`: Projection embedding size used for the contrastive space after the
  encoders. Tune this when comparing contrastive representation capacity.
- `--tau`: Temperature for the contrastive softmax. Lower values emphasize the
  strongest similarities; higher values smooth the similarity distribution.
- `--lr`: RMSprop learning rate. Lower it if the loss is unstable; raise it
  cautiously for faster learning.
- `--epochs`: Number of training epochs. Use small values for smoke tests and
  larger values for full runs.
- `--disable-cuda`: Forces CPU training even when CUDA is available. Use this for
  debugging device issues or reproducing CPU-only behavior.
- `--log-every-n-steps`: Epoch interval for computing and logging training metrics
  to W&B. Larger intervals reduce logging overhead.
- `--use_hpo_embeddings`: Use `1` to initialize HPO nodes from the sentence
  embedding file, or `0` to fall back to one-hot HPO features.
- `--concat_hpo_embeddings`: Use `1` to concatenate original HPO sentence
  embeddings onto learned HPO embeddings during full-dataset validation.
- `--hpo_embeddings_path`: Path to the `.npy` HPO sentence embedding file. Change
  this when using a different embedding model or local data layout.
- `--wandb_label`: Name for the offline W&B run and the label appended to saved
  validation artifacts. Use descriptive labels when comparing experiments.
- `--encoder_mode`: Selects the disease representation used for evaluation:
  `hpo` pools HPO embeddings, `hnet` pools gene-network embeddings, and
  `combined` evaluates both views together.
- `--full_dataset`: Disease-pair validation file used after training. Point this
  to the held-out test set for final evaluation.
- `--beta`: Weight for the bidirectional contrastive loss. `beta` weights the HPO
  direction and `1 - beta` weights the gene direction.
- `--gamma`: Combined-mode validation weight for balancing HPO and gene disease
  embeddings. This only matters with `--encoder_mode combined`.
- `--hyperparameter_tuning`: Runs the Optuna tuning workflow instead of regular
  training, then exits after saving the best parameters.
- `--cv_folds`: Number of cross-validation folds used during hyperparameter
  tuning. More folds give a more stable estimate but take longer.
- `--tuning_dataset`: Dataset file used to build cross-validation folds during
  hyperparameter tuning. This should normally be a training split, not the final
  held-out test set.
- `--n_trials`: Number of Optuna trials. Increase for a broader search; decrease
  for quick tuning checks.
- `--output_dir`: Directory where hyperparameter tuning artifacts are saved,
  including best parameters, plots, trial summaries, and study metadata.

## Citation

If you use this code or our work, please cite the paper:

```
@misc{baminiwatte2025phenognet,
title={PhenoGnet: A Graph-Based Contrastive Learning Framework for Disease Similarity Prediction},
author={Ranga Baminiwatte and Kazi Jewel Rana and Aaron J. Masino},
year={2025},
eprint={2509.14037},
archivePrefix={arXiv},
primaryClass={q-bio.GN}
}
```

## Acknowledgments

This work was supported by the NIH funded Center of Biomedical Research Excellence in Human Genetics at Clemson University.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
