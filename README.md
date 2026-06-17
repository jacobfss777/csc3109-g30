# CSC3109 Machine Learning — Group 30

AY2025/2026 Trimester 3 | Aerial Scene Classification

## Project Overview

This project builds an image-classification system that identifies aerial photographs into one of four energy/industrial infrastructure categories:

- **oil_well** — circular drilling pad structures visible from above
- **solar_panel** — large rectangular arrays of photovoltaic panels
- **transformer_station** — electrical substation grid structures
- **wastewater_treatment_plant** — circular settling tanks and processing pools

These categories are visually confusable from the air, making fine-grained classification the core challenge.

---

## Dataset

| Split | Images per class | Total |
|-------|-----------------|-------|
| Train | 700 | 2,800 |
| Val   | 100 | 400   |

Raw training data is located in `ml training data/set 30/`. The validation split is held out and used only for evaluation — never for training.

---

## Project Structure

```
csc3109-ml-assignment/
├── ml training data/
│   └── set 30/
│       ├── oil_well/                  # 700 training images
│       ├── solar_panel/               # 700 training images
│       ├── transformer_station/       # 700 training images
│       └── wastewater_treatment_plant/# 700 training images
├── data/
│   ├── train/                         # symlinked/copied training split
│   └── val/                           # held-out validation split
├── notebooks/                         # Jupyter notebooks (EDA, experiments)
├── models/                            # saved model checkpoints (.pth files)
├── src/                               # Python source code
│   ├── dataset.py                     # DataLoader and transforms
│   ├── train.py                       # training loop
│   ├── evaluate.py                    # metrics and evaluation
│   └── app.py                         # FastAPI inference server
├── docker/
│   └── Dockerfile                     # container definition
├── report/                            # report drafts
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Environment Setup

### Prerequisites

Install the following tools before running any code. Each link goes to the official download page.

#### 1. Python 3.10 or 3.11

Python is the programming language used throughout this project.

- Download: https://www.python.org/downloads/
- During installation on Windows, **check "Add Python to PATH"**
- Verify after install:
  ```bash
  python --version
  # Expected: Python 3.10.x or 3.11.x
  ```

#### 2. Git

Version control — already installed if you cloned this repo. Verify:
```bash
git --version
```

#### 3. Visual Studio Code (VS Code)

Code editor with Jupyter notebook support.

- Download: https://code.visualstudio.com/
- After installing VS Code, install these extensions from the Extensions panel (Ctrl+Shift+X):
  - **Python** (by Microsoft)
  - **Jupyter** (by Microsoft)

#### 4. Docker Desktop

Used in the final phase to package and deploy the model as a portable container.

- Download: https://www.docker.com/products/docker-desktop/
- Windows requires WSL 2 backend (the installer will guide you)
- Verify after install:
  ```bash
  docker --version
  # Expected: Docker version 24.x.x or higher
  ```

---

## Python Environment Setup

It is best practice to use a **virtual environment** so that packages for this project do not interfere with other Python projects on your machine.

### Step 1 — Clone the repository (if not done already)

```bash
git clone https://github.com/jacobfss777/csc3109-g30.git
cd csc3109-g30
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv

# macOS / Linux
python3 -m venv venv
```

This creates a folder called `venv/` containing an isolated Python installation.

### Step 3 — Activate the virtual environment

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

You will see `(venv)` prepended to your terminal prompt, confirming the environment is active.

### Step 4 — Install PyTorch

PyTorch must be installed separately from the other requirements because the correct version depends on whether you have an NVIDIA GPU.

**Option A — CPU only (works on any machine, slower training)**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Option B — NVIDIA GPU (CUDA 12.6, much faster training)**

Only use this if your machine has an NVIDIA GPU. Check by running `nvidia-smi` in your terminal.
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output (CPU):
```
PyTorch: 2.6.0
CUDA available: False
```

Expected output (GPU):
```
PyTorch: 2.6.0
CUDA available: True
```

### Step 7 — Register the virtual environment as a Jupyter kernel

This lets Jupyter notebooks use the packages you just installed.
```bash
python -m ipykernel install --user --name=csc3109 --display-name "CSC3109 ML"
```

### Step 8 — Launch Jupyter Notebook

```bash
jupyter notebook
```

A browser window will open. Navigate to the `notebooks/` folder to open any notebook. When prompted to select a kernel, choose **CSC3109 ML**.

---

## Deep Learning Approaches

Five approaches are investigated and compared:

| # | Approach | Architecture | Strategy |
|---|----------|-------------|----------|
| 1 | Custom CNN | Built from scratch | Baseline — trained from random weights |
| 2 | ResNet-50 Feature Extraction | ResNet-50 (ImageNet) | All layers frozen, only classifier head trained |
| 3 | ResNet-50 Fine-Tuning | ResNet-50 (ImageNet) | Later layers unfrozen, trained with low learning rate |
| 4 | EfficientNet-B0 Fine-Tuning | EfficientNet-B0 (ImageNet) | Full fine-tune of modern efficient architecture |
| 5 | Vision Transformer (ViT-B/16) | ViT-B/16 (ImageNet-21k) | Attention-based architecture, fine-tuned |

---

## Running Training

```bash
# Activate environment first
.\venv\Scripts\Activate.ps1

# Train a specific approach (1–5)
python src/train.py --approach 1
python src/train.py --approach 2
# etc.
```

Trained model weights are saved to `models/`.

---

## Running Evaluation

```bash
python src/evaluate.py --model models/best_model.pth
```

Outputs accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Docker Deployment

```bash
# Build the container image
docker build -t aerial-classifier -f docker/Dockerfile .

# Run the inference server
docker run -p 8000:8000 aerial-classifier

# Send an image for classification
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/image.jpg"

# Expected response
# {"label": "solar_panel", "confidence": 0.9712}
```

---

## Evaluation Metrics

All models are evaluated on the held-out validation set using:

- **Accuracy** — overall correct predictions / total predictions
- **Precision** — per-class: true positives / (true positives + false positives)
- **Recall** — per-class: true positives / (true positives + false negatives)
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — 4×4 heatmap showing where misclassifications occur

---

## Team

| Member | Responsibility |
|--------|---------------|
| Jacob  | Project lead, Custom CNN (Approach 1) |
| TBD    | ResNet-50 Feature Extraction (Approach 2) |
| TBD    | ResNet-50 Fine-Tuning (Approach 3) |
| TBD    | EfficientNet-B0 (Approach 4) + Docker |
| TBD    | Vision Transformer (Approach 5) + Report |

---

## Submission

- **Deadline:** Sunday, 26 July 2026, 23:59 (local time)
- `T30.pdf` — final report
- `T30.zip` — video demonstration and supplementary materials
