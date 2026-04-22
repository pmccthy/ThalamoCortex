# ThalamoCortex

> PyTorch models of cortico-thalamo-cortical circuits for studying the computational role of higher-order thalamic projections in hierarchical sensory processing.

---

## Overview

The thalamus is far more than a simple sensory relay. **Higher-order thalamic nuclei** receive driver input from cortical layer 5 and project back to superficial cortical layers, forming loops that may play a key role in hierarchical inference, attention, and the routing of information across cortical areas.

This repository provides a modular PyTorch library — `thalamocortex` — for building and training **Cortico-Thalamo-Cortical (CTC) networks**, alongside training scripts, analysis notebooks, and custom dataset utilities.

### Key features

- Unified `CTCNet` class that supports a range of thalamocortical interaction mechanisms via a single parameter
- Two-pass forward dynamics: thalamic activity is first estimated from cortical states, then fed back on the second pass
- Optional per-layer thalamic areas (one thalamic nucleus per cortical area) or a single pooled thalamic area
- Configurable reciprocal thalamocortical projections and thalamic-to-readout connections
- `CTCNetThalReadout` variant with an explicit thalamic readout head for probing thalamic representations
- Training utilities with top-k accuracy, grid search, and [Weights & Biases](https://wandb.ai) logging
- Custom datasets: LeftRight MNIST and Binary MNIST

---

## Architecture

The CTC network consists of two cortical layers (`ctx1`, `ctx2`), a thalamic area (`thal`), and a linear readout. The forward pass runs in two steps:

```
Step 1 — initialise thalamus to zero, run subforward to estimate thalamic activity:

  input ──► ctx1 ──► ctx2 ──► thal
                                │
                         (thal estimated)

Step 2 — run subforward again with thalamic feedback:

  input ──► ctx1 ──► ctx2 ──► readout ──► output
     ▲        ▲        ▲
     └────────┴────────┴──── thal (feedback projections)
```

The nature of the thalamic feedback is controlled by `thalamocortical_type`:

| Type | Mechanism | Description |
|---|---|---|
| `None` | — | Purely feedforward baseline (no thalamus) |
| `add` | `ctx = f(x + W_thal · thal)` | Additive thalamic drive before summation |
| `multi_pre_sum` | `ctx = f(x * W_thal · thal)` | Multiplicative gating before summation |
| `multi_pre_activation` | `ctx = relu((Wx) * W_thal · thal)` | Multiplicative gating after linear sum, before ReLU |
| `multi_post_activation` | `ctx = f(x) * W_thal · thal` | Multiplicative gating after activation |

---

## Installation

Requires Python ≥ 3.10. It is strongly recommended to use a Conda environment.

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/ThalamoCortex.git
cd ThalamoCortex
```

**2. Create and activate the Conda environment**

```bash
conda env create -f env.yml
conda activate burstccn
```

**3. Install the package in editable mode**

```bash
pip install -e .
```

---

## Quick Start

```python
from thalamocortex.models import CTCNet

model = CTCNet(
    input_size=784,          # e.g. flattened 28×28 MNIST image
    output_size=10,          # number of classes
    ctx_layer_size=256,      # cortical layer width
    thal_layer_size=128,     # thalamic layer width
    thalamocortical_type="multi_pre_activation",  # feedback mechanism
    thal_reciprocal=True,    # thalamus projects back to cortical layers
    thal_to_readout=True,    # thalamus also projects to readout layer
    thal_per_layer=False,    # single pooled thalamic area
)

import torch
x = torch.randn(32, 1, 28, 28)   # batch of 32 MNIST images
logits = model(x)                  # shape: (32, 10)
```

To use the model with an explicit thalamic readout head (for probing thalamic representations):

```python
from thalamocortex.models import CTCNetThalReadout

model = CTCNetThalReadout(
    input_size=784,
    ctx_output_size=10,
    thal_output_size=10,
    ctx_layer_size=256,
    thal_layer_size=128,
    thalamocortical_type="add",
)

ctx_logits, thal_logits = model(x)
```

---

## Repository Structure

```
ThalamoCortex/
├── thalamocortex/              # Core library
│   ├── models.py               # CTCNet, CTCNetThalReadout, CortexWithThalamicMultiPreAct
│   └── utils.py                # Data loaders, train/eval loops, W&B logging, grid search
├── scripts/                    # Python training and analysis scripts
│   ├── train_driver*.py        # Scripts for additive / driver-type thalamus
│   ├── train_mod*.py           # Scripts for multiplicative / modulator-type thalamus
│   ├── train_feedforward*.py   # Purely feedforward baselines
│   ├── train_ff_finetune.py    # Fine-tuning with thalamic readout
│   ├── leftright_mnist_analysis.py
│   └── shell/                  # Batch shell scripts for cluster/grid runs
│       └── *.sh
├── notebooks/
│   ├── training/               # Interactive training and prototyping notebooks
│   └── analysis/               # Post-hoc analysis and interpretation notebooks
├── data/                       # Dataset generation notebooks
│   ├── generate_binarymnist.ipynb
│   └── generate_leftrightmnist.ipynb
├── env.yml                     # Conda environment
├── setup.py
└── requirements.txt
```

---

## Training

Training scripts are in `scripts/`. Each script exposes a grid-search over hyperparameters with optional W&B logging.

**Example — train a multiplicative pre-activation model on MNIST:**

```bash
python scripts/train_mod1_mnist.py
```

**Example — run a batch of experiments with a shell script:**

```bash
bash scripts/shell/train_drivers_mods.sh
```

Results and checkpoints are saved locally; to enable W&B logging, set your API key:

```bash
wandb login
```

---

## Datasets

The following datasets are supported out of the box via the utilities in `thalamocortex/utils.py`:

| Dataset | Description |
|---|---|
| MNIST | Standard handwritten digits |
| FashionMNIST | Zalando fashion article images |
| CIFAR-10 | 32×32 colour images across 10 classes |
| BinaryMNIST | Two-class MNIST variant (generated via `data/generate_binarymnist.ipynb`) |
| LeftRightMNIST | Spatially-biased MNIST variant for lateralisation experiments (generated via `data/generate_leftrightmnist.ipynb`) |

---

## Analysis

Notebooks for post-hoc analysis of trained models are in `notebooks/analysis/`, with supporting scripts in `scripts/`. These include:

- Representational analysis of thalamic and cortical activations
- Fine-tuning experiments
- Model interpretation and receptive field visualisation
- Inference on held-out data

---

## Requirements

- Python ≥ 3.10
- PyTorch 1.12
- torchvision 0.13
- wandb
- scikit-learn
- torchsummary
- jupyter

See `env.yml` for the full pinned environment.

---

## Author

**Patrick McCarthy**  
DTC, University of Oxford  
[patrick.mccarthy@dtc.ox.ac.uk](mailto:patrick.mccarthy@dtc.ox.ac.uk)

---

## Licence

This project is licensed under the MIT Licence. See `LICENSE` for details.
