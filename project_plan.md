# CSC3109 ML Group Project — Complete Plan

**Team:** Group 30 (5 members)  
**Dataset:** 4 aerial imagery categories (oil_well, solar_panel, transformer_station, wastewater_treatment_plant)  
**Deadline:** Sunday, 26 July 2026, 23:59 (local time)  
**Submission:** T30.pdf (report) + T30.zip (video + supplementary materials)

---

## Phase 0 — Project Setup and Environment Configuration ✅

**Status:** Completed (20 June 2026)

### Objectives
- Set up development environment for all team members
- Organise dataset and folder structure
- Install all required libraries and tools

### Deliverables
- ✅ Python 3.13 virtual environment (`venv/`)
- ✅ `requirements.txt` with all dependencies (PyTorch 2.6.0, timm, scikit-learn, jupyter, fastapi, etc.)
- ✅ Project folder structure (`notebooks/`, `models/`, `src/`, `docker/`, `report/`, `data/`)
- ✅ Comprehensive `README.md` with installation instructions
- ✅ `.gitignore` to exclude venv, model weights, and cache
- ✅ Jupyter kernel registered ("CSC3109 ML")
- ✅ Dataset verified: 4 categories × 700 images = 2,800 training images

### Team Assignment
- **Jacob:** Project coordination, repository setup, environment configuration

### How Team Members Can Get Started
Each team member runs the following after cloning the repo:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install PyTorch (CPU)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=csc3109 --display-name "CSC3109 ML"

# Start Jupyter
jupyter notebook
```

---

## Phase 1 — Exploratory Data Analysis (EDA)

**Timeline:** 21 June – 30 June 2026  
**Duration:** ~2 weeks (part-time)

### Objectives
- Understand dataset characteristics before training
- Identify class imbalance, image distribution, and visual confusability
- Establish baseline observations for the report

### Steps

#### Step 1.1 — Data Loading and Counting
- Load all images from `ml training data/set 30/`
- Count images per category (confirm 700 per class)
- Check for corrupted files or missing images
- List image dimensions and file sizes

#### Step 1.2 — Visual Inspection
- Display sample images (4–8) from each category
- Arrange in a grid for side-by-side comparison
- Observe visual similarities and differences
- Document which categories are hardest to distinguish

#### Step 1.3 — Pixel Value Analysis
- Plot histogram of pixel values across all images
- Compute mean and standard deviation per channel (R, G, B)
- Check for outliers or unusual distributions
- Visualise these statistics per category

#### Step 1.4 — Image Metadata
- Verify all images are same resolution
- Check colour spaces (RGB, grayscale, etc.)
- Document any preprocessing observations

#### Step 1.5 — Create Validation Split
- Randomly split 100 images per category into `data/val/`
- Keep remaining 700 per category in `data/train/`
- Ensure no overlap (validation must be held-out)
- Document the split process and random seed

### Deliverables
- **Jupyter Notebook:** `notebooks/01_eda.ipynb`
  - All plots, statistics, and observations
  - Clear markdown documentation
  - Code should be reusable and well-commented
- **Summary Report Section:** Draft 2–3 pages for the final report covering:
  - Dataset overview
  - Visual analysis of confusability
  - Pixel statistics and preprocessing observations

### Visualisations to Create
- Grid of sample images (4 per category)
- Histogram of pixel values
- Mean pixel intensity per category
- Class distribution bar chart

### Team Assignment
- **Jacob:** Lead EDA, create notebook, document findings
- **Other members:** Review findings, provide feedback on observations

---

## Phase 2 — Data Preparation and Augmentation

**Timeline:** 1 July – 7 July 2026  
**Duration:** ~1 week

### Objectives
- Prepare images in a format suitable for neural networks
- Implement data augmentation to increase training diversity
- Set up PyTorch DataLoaders for efficient batch training

### Steps

#### Step 2.1 — Normalisation
- Resize all images to 224×224 (standard for transfer learning)
- Convert to tensors (PyTorch format)
- Normalise using ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Important:** Apply normalization to training AND validation data

#### Step 2.2 — Training Augmentation
- Implement augmentation pipeline applied ONLY to training set:
  - Random horizontal flip (50% probability)
  - Random rotation (±15 degrees)
  - Random colour jitter (brightness, contrast, saturation ±0.2)
  - Optional: random crop, affine transforms
- Document rationale for each augmentation choice

#### Step 2.3 — Validation Transform
- Resize and normalise ONLY (no augmentation)
- Ensures fair evaluation on unmodified images

#### Step 2.4 — PyTorch DataLoaders
- Create `ImageDataset` class that loads images on-the-fly
- Implement `train_loader` (batch size 32, shuffled)
- Implement `val_loader` (batch size 32, not shuffled)
- Test data loading: verify batches are loaded correctly

### Deliverables
- **Python Module:** `src/dataset.py`
  - `ImageDataset` class with transforms
  - Factory functions `get_train_loader()` and `get_val_loader()`
  - Example usage in docstring
- **Jupyter Notebook:** `notebooks/02_data_prep.ipynb`
  - Visualise augmentation effects (show same image with different augmentations)
  - Verify DataLoaders work correctly
  - Display sample batches

### Key Concepts to Document
- Why normalisation improves training (gradient flow, convergence)
- Why augmentation helps (prevents overfitting, increases effective dataset size)
- Why validation data is not augmented (fair evaluation)

### Team Assignment
- **Jacob or Member 2:** Implement `dataset.py`
- **Team:** Review and test DataLoaders

---

## Phase 3 — Deep Learning Approaches (Core Work)

**Timeline:** 8 July – 18 July 2026  
**Duration:** ~2 weeks (intensive)

### Objectives
- Implement 5 distinct deep learning approaches
- Train and validate each model
- Compare performance across approaches

### Approach 1 — Custom CNN from Scratch

**Responsible Member:** Jacob  
**Architecture:**
```
Input (224×224×3)
→ Conv2D(32, 3×3) + ReLU + MaxPool(2×2)
→ Conv2D(64, 3×3) + ReLU + MaxPool(2×2)
→ Conv2D(128, 3×3) + ReLU + MaxPool(2×2)
→ Flatten
→ Linear(256) + Dropout(0.5) + ReLU
→ Linear(4)  ← output logits for 4 classes
→ Softmax
```

**Steps:**
1. Implement model in `src/models/custom_cnn.py`
2. Write training loop in `src/train.py`
3. Train for 50 epochs, save checkpoint with best validation accuracy
4. Document training curve (loss + accuracy)
5. Create notebook `notebooks/03a_custom_cnn.ipynb` with experiments

**Key Metrics:**
- Training loss curve
- Validation accuracy curve
- Training time

---

### Approach 2 — ResNet-50 Feature Extraction

**Responsible Member:** Member 2  
**Strategy:** Freeze all ResNet-50 layers (pre-trained on ImageNet), train only the classification head

**Steps:**
1. Load pre-trained ResNet-50 from torchvision
2. Freeze all layers: `for param in model.parameters(): param.requires_grad = False`
3. Replace final FC layer with `Linear(2048, 4)` — this is the only trainable layer
4. Train for 30 epochs with learning rate 1e-3
5. Save best model checkpoint
6. Create notebook `notebooks/03b_resnet50_extraction.ipynb`

**Key Concepts:**
- Transfer learning: reuse learned features from ImageNet
- Feature extraction: only adapt the classifier, not the features
- Why this works: aerial images share low-level features (edges, textures) with ImageNet

---

### Approach 3 — ResNet-50 Fine-Tuning

**Responsible Member:** Member 3  
**Strategy:** Unfreeze later layers of ResNet-50 and train them with a lower learning rate

**Steps:**
1. Start from best checkpoint of Approach 2
2. Unfreeze `layer4` (the last residual block) and the FC layer
3. Use differential learning rates:
   - `layer4` parameters: learning rate 1e-5
   - FC parameters: learning rate 1e-4
4. Train for 20 epochs
5. Save best model checkpoint
6. Create notebook `notebooks/03c_resnet50_finetuning.ipynb`

**Key Concepts:**
- Fine-tuning: domain-specific adaptation of pre-trained features
- Differential learning rates: lower rates for deeper layers (they already learned useful features)
- Expected improvement: Fine-tuning typically outperforms feature extraction for domain shift

---

### Approach 4 — EfficientNet-B0 Fine-Tuning

**Responsible Member:** Member 4  
**Architecture:** EfficientNet-B0 (more efficient than ResNet at same accuracy)

**Steps:**
1. Load pre-trained EfficientNet-B0 from torchvision
2. Replace classifier head: `model.classifier[1] = Linear(1280, 4)`
3. Unfreeze all layers (or just the last 2 blocks) for fine-tuning
4. Use learning rate 5e-5
5. Train for 40 epochs
6. Save best model checkpoint
7. Create notebook `notebooks/03d_efficientnet.ipynb`

**Key Concepts:**
- EfficientNet: modern architecture with better parameter efficiency
- Scaling: depth, width, resolution scaled together
- Likely candidate for best-performing model

---

### Approach 5 — Vision Transformer (ViT-B/16) Fine-Tuning

**Responsible Member:** Member 5  
**Architecture:** Vision Transformer (ViT-B/16) — attention-based instead of convolutions

**Steps:**
1. Install `timm`: `pip install timm`
2. Load pre-trained ViT-B/16 from ImageNet-21k
3. Replace head for 4-class classification
4. Fine-tune with learning rate 1e-4 for 50 epochs
5. Save best model checkpoint
6. Create notebook `notebooks/03e_vit.ipynb`

**Key Concepts:**
- Vision Transformer: patch-based attention, captures global context
- State-of-the-art direction in computer vision (2022+)
- May work particularly well on aerial imagery (global structure matters)

---

### Phase 3 Common Workflow

**For each approach:**

1. **Training Setup**
   - Define model architecture
   - Set loss function: `nn.CrossEntropyLoss()`
   - Choose optimiser: Adam or SGD
   - Optional: learning rate scheduler (ReduceLROnPlateau)

2. **Training Loop**
   ```python
   for epoch in range(num_epochs):
       model.train()
       for batch_images, batch_labels in train_loader:
           optimizer.zero_grad()
           outputs = model(batch_images)
           loss = criterion(outputs, batch_labels)
           loss.backward()
           optimizer.step()
       
       # Validate
       model.eval()
       val_loss = 0
       val_acc = 0
       with torch.no_grad():
           for images, labels in val_loader:
               outputs = model(images)
               loss = criterion(outputs, labels)
               val_loss += loss.item()
               # compute accuracy
       
       # Save best model
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), f'models/{approach_name}_best.pth')
   ```

3. **Documentation**
   - Plot training/validation curves
   - Document hyperparameter choices and rationale
   - Note training time and GPU/CPU used
   - List any difficulties encountered

### Deliverables
- 5 trained model checkpoints: `models/approach_{1,2,3,4,5}_best.pth`
- 5 Jupyter notebooks: `notebooks/03{a,b,c,d,e}_*.ipynb`
- 15–20 pages for the report covering:
  - Architecture diagrams for each approach
  - Training curves (loss + accuracy)
  - Hyperparameter choices and rationale
  - Qualitative observations during training

### Team Coordination
- Each member works on their assigned approach in parallel
- Daily standup (15 min) to share progress and blockers
- Weekly code review on main branch before merging

---

## Phase 4 — Hyperparameter Tuning

**Timeline:** 15 July – 18 July 2026  
**Duration:** ~1 week (concurrent with Phase 3)

### Objectives
- Systematically improve the best model(s)
- Identify optimal hyperparameters for your dataset

### Steps

#### Step 4.1 — Grid Search on Best Approach
- After all 5 approaches are trained, identify the top 2 performers
- For the best approach, create a hyperparameter grid:
  - Learning rates: [1e-5, 5e-5, 1e-4, 5e-4]
  - Batch sizes: [16, 32, 64]
  - Augmentation strength: [mild, moderate, aggressive]
  - Dropout rates: [0.3, 0.5, 0.7]

#### Step 4.2 — Systematic Evaluation
- Train ~10–15 combinations
- Log validation accuracy for each combination
- Create comparison table: hyperparameters → validation accuracy

#### Step 4.3 — Learning Rate Scheduling
- Experiment with learning rate decay strategies:
  - Step decay: reduce LR by 0.1 every N epochs
  - Cosine annealing: smooth decay from initial to minimum LR
  - ReduceLROnPlateau: reduce LR when validation plateaus

### Deliverables
- **Hyperparameter comparison table:** CSV with combinations and results
- **Notebook:** `notebooks/04_hyperparameter_tuning.ipynb`
- **Best model checkpoint:** `models/best_model_tuned.pth`

### Team Assignment
- **Jacob + Member 4:** Lead tuning on best approach
- **Other members:** Assist with running experiments

---

## Phase 5 — Performance Evaluation

**Timeline:** 19 July – 21 July 2026  
**Duration:** ~3 days

### Objectives
- Compute comprehensive evaluation metrics
- Compare all 5 approaches
- Generate visualisations for the report

### Steps

#### Step 5.1 — Inference on Validation Set
- Load best checkpoint for each of the 5 approaches
- Run inference on all 400 validation images (100 per class)
- Collect predicted labels and confidence scores

#### Step 5.2 — Metrics Computation
For each model, compute:

| Metric | Definition |
|--------|-----------|
| **Accuracy** | # correct / # total |
| **Precision (per class)** | TP / (TP + FP) |
| **Recall (per class)** | TP / (TP + FN) |
| **F1-score (per class)** | 2 × (Precision × Recall) / (Precision + Recall) |
| **Confusion Matrix** | 4×4 grid showing predicted vs actual |

Use `sklearn.metrics.classification_report()` and `confusion_matrix()`

#### Step 5.3 — Comparative Analysis
- Create a summary table:
  ```
  | Approach | Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
  |----------|----------|-----------------|--------------|----------------|
  | Custom CNN | X.XX | X.XX | X.XX | X.XX |
  | ResNet50 Extraction | X.XX | X.XX | X.XX | X.XX |
  | ResNet50 Fine-tune | X.XX | X.XX | X.XX | X.XX |
  | EfficientNet | X.XX | X.XX | X.XX | X.XX |
  | ViT | X.XX | X.XX | X.XX | X.XX |
  ```

#### Step 5.4 — Visualisations
- Confusion matrices (heatmaps) for each approach
- Accuracy comparison bar chart
- Per-class precision/recall radar charts
- Training vs validation curves for all approaches

#### Step 5.5 — Error Analysis
- Identify most commonly confused category pairs
- Display misclassified examples (with predicted vs actual labels)
- Analyse whether errors are systematic (e.g., always confuse A with B)

### Deliverables
- **Notebook:** `notebooks/05_evaluation.ipynb`
- **Metrics summary table:** CSV or markdown table
- **Plots:** confusion matrices, accuracy comparison, error analysis
- **Report section:** 3–4 pages with results, insights, and discussion of strengths/weaknesses

### Team Assignment
- **Jacob:** Coordinate evaluation, create summary tables
- **Each member:** Write strengths/weaknesses section for their approach

---

## Phase 6 — Docker Containerisation and Deployment

**Timeline:** 22 July – 24 July 2026  
**Duration:** ~2 days

### Objectives
- Package best model as a portable, self-contained application
- Create inference endpoint that accepts images and returns predictions

### Steps

#### Step 6.1 — FastAPI Inference Server
Create `src/app.py`:
```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import io

app = FastAPI()
model = torch.load("models/best_model_tuned.pth")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess
    transform = ...  # same as validation transform
    image_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        confidence, predicted_class = torch.max(logits, 1)
    
    class_names = ["oil_well", "solar_panel", "transformer_station", "wastewater_treatment_plant"]
    
    return {
        "label": class_names[predicted_class.item()],
        "confidence": confidence.item(),
        "logits": logits.tolist()
    }
```

#### Step 6.2 — Dockerfile
Create `docker/Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code and model
COPY src/ src/
COPY models/best_model_tuned.pth models/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Step 6.3 — Build and Test
```bash
docker build -t aerial-classifier -f docker/Dockerfile .
docker run -p 8000:8000 aerial-classifier
```

Test with curl:
```bash
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"
```

#### Step 6.4 — Documentation
- Document API endpoints in `docker/README.md`
- Provide example curl commands
- List expected input/output formats

### Deliverables
- **`docker/Dockerfile`:** Container definition
- **`src/app.py`:** FastAPI server
- **`docker/README.md`:** Usage instructions and examples
- **Report section:** 1–2 pages on containerisation strategy

### Team Assignment
- **Member 4:** Lead Docker implementation and testing

---

## Phase 7 — Report Writing and Final Submission

**Timeline:** 15 July – 26 July 2026  
**Duration:** ~2 weeks (concurrent with Phases 4–6)

### Report Structure (30 pages target, max 50)

#### Section 1: Overall Project Description (3–6 pages)
- **Background and Motivations**
  - Why aerial imagery classification matters
  - Real-world applications (urban planning, environmental monitoring, disaster response)
  - Motivation for the specific 4-category problem

- **Problem Statement**
  - Define the 4 categories (oil_well, solar_panel, transformer_station, wastewater_treatment_plant)
  - Explain visual confusability — why these categories are hard to distinguish
  - Challenge: fine-grained classification in a small training set (700 images per class)

- **Project Objectives**
  - Build end-to-end ML pipeline
  - Investigate 5 different deep learning approaches
  - Achieve best possible accuracy on held-out validation set
  - Package solution as Docker container

- **Literature Review (2–3 pages)**
  - Summarise 5–8 key papers:
    - ResNet (He et al., 2015) — skip connections
    - EfficientNet (Tan & Le, 2019) — efficient scaling
    - Vision Transformer (Dosovitskij et al., 2020) — attention for vision
    - Transfer Learning surveys
    - Domain Adaptation papers
  - Discuss relevance to your problem

#### Section 2: Machine Learning Solutions (15–18 pages)

- **Exploratory Data Analysis (2 pages)**
  - Dataset statistics (image counts, dimensions, file sizes)
  - Visual examples from each category
  - Pixel value distributions
  - Class balance analysis

- **Data Preparation and Pre-processing (2 pages)**
  - Normalisation strategy (ImageNet statistics)
  - Data augmentation pipeline (list and justify each transform)
  - Train/validation split methodology
  - DataLoader batch size and shuffling strategy

- **Investigation of Deep Learning Approaches (8 pages, 1.5 per approach)**
  - **Approach 1 — Custom CNN:** Architecture diagram, rationale, training curve
  - **Approach 2 — ResNet-50 Feature Extraction:** Why transfer learning, training details
  - **Approach 3 — ResNet-50 Fine-tuning:** Rationale for unfreezing layers
  - **Approach 4 — EfficientNet-B0:** Modern efficient architecture motivation
  - **Approach 5 — Vision Transformer:** Attention mechanism, why for aerial imagery
  - For each: architecture details, hyperparameters, training time, observations

- **Model Training and Tuning (2 pages)**
  - Loss function choice (CrossEntropyLoss)
  - Optimiser selection (Adam vs SGD)
  - Learning rate scheduling strategy
  - Hyperparameter grid search results
  - Convergence analysis

- **Deep Learning Inference (1 page)**
  - Batch inference implementation
  - Inference time per image
  - Memory requirements

- **Performance Evaluation (2 pages)**
  - Metrics: accuracy, precision, recall, F1-score
  - Comparison table (all 5 approaches)
  - Confusion matrices for top 2 approaches
  - Per-class performance analysis

- **Strengths and Weaknesses (1 page)**
  - Best approach: what makes it effective?
  - Weaknesses: where does it fail?
  - Failure case analysis with examples

- **Containerisation (1 page)**
  - Docker strategy and rationale
  - API endpoint design
  - Deployment instructions

#### Section 3: Individual Contributions (1–2 pages)
- **Jacob:** Project lead, Phase 0 setup, Phase 1 EDA, Phase 3a Custom CNN, Phase 4 tuning, report coordination
- **Member 2:** Phase 3b ResNet-50 Feature Extraction, evaluation support
- **Member 3:** Phase 3c ResNet-50 Fine-tuning, report section
- **Member 4:** Phase 3d EfficientNet, Phase 6 Docker, deployment docs
- **Member 5:** Phase 3e Vision Transformer, Phase 7 report compilation

#### Section 4: Reflection (2–3 pages)
- What you learned about deep learning
- Practical insights from implementing 5 approaches
- Challenges encountered and how you overcame them
- How the theory from lectures applied to this real-world problem

#### Section 5: Future Work (1–2 pages)
- Larger Vision Transformers (ViT-L, ViT-H)
- Self-supervised learning (contrastive, masked image modelling)
- Ensemble methods (combining top 2–3 approaches)
- Data collection: more training images per class
- Fine-grained localisation: which parts of image are diagnostic?
- Multi-scale analysis: training on different image crops/resolutions

### Deliverables
- **T30.pdf** — final report (30–50 pages)
- **T30.zip** — compressed file containing:
  - 3–5 minute video demonstration
  - Best model checkpoint
  - Supplementary notebooks
  - Docker build scripts

### Video Demonstration (3–5 minutes)
- **Intro** (30 sec): team members introduce the project
- **Motivation** (30 sec): why aerial image classification matters
- **Approaches** (1 min): brief summary of 5 approaches
- **Results** (1 min): show confusion matrices, accuracy comparison
- **Demo** (1 min): run inference on real test images, show predictions + confidence
- **Conclusion** (30 sec): key learnings

### Team Assignment
- **Jacob:** Coordinate report writing, write Sections 1 & 3
- **Member 2:** Write Approach 2 subsection + 2 pages literature review
- **Member 3:** Write Approach 3 subsection + 2 pages literature review
- **Member 4:** Write Approach 4 subsection + containerisation section
- **Member 5:** Write Approach 5 subsection + Section 4 & 5
- **All members:** Review, provide feedback, iterate on drafts

---

## Key Dates and Milestones

| Date | Milestone | Responsible |
|------|-----------|-------------|
| 20 Jun ✅ | Phase 0 complete (setup) | Jacob |
| 30 Jun | Phase 1 complete (EDA) | Jacob |
| 7 Jul | Phase 2 complete (data prep) | Jacob / Member 2 |
| 18 Jul | Phase 3 complete (all 5 approaches trained) | All members |
| 18 Jul | Phase 4 complete (hyperparameter tuning) | Jacob / Member 4 |
| 21 Jul | Phase 5 complete (evaluation) | Jacob |
| 24 Jul | Phase 6 complete (Docker deployment) | Member 4 |
| 25 Jul | Phase 7 draft 1 (full report) | All members |
| 26 Jul 23:59 | **FINAL SUBMISSION** | All members |

---

## How to Track Progress

- Update this file weekly with actual completion dates
- Use git commits to mark phase completions
- Daily 15-min standup for coordination
- Weekly code reviews before merging to main

---

## Resources

| Topic | Resource |
|-------|----------|
| PyTorch basics | https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html |
| Fast.ai course | https://course.fast.ai/ |
| Neural network intuition | https://www.youtube.com/watch?v=aircAruvnKk (3Blue1Brown) |
| ResNet paper | https://arxiv.org/abs/1512.03385 |
| EfficientNet paper | https://arxiv.org/abs/1905.11946 |
| Vision Transformer paper | https://arxiv.org/abs/2010.11929 |
| Transfer Learning | https://cs231n.github.io/transfer-learning/ |

---

## Submission Checklist

- [ ] Report PDF (T30.pdf) is complete, proofread, 30–50 pages
- [ ] Video demonstration (3–5 min) is recorded and included in T30.zip
- [ ] All 5 model checkpoints trained and evaluated
- [ ] Docker container builds and runs successfully
- [ ] Inference endpoint responds to test requests
- [ ] Confusion matrices and metric tables included in report
- [ ] Individual contributions section lists all members and their work
- [ ] Literature review includes 5–8 papers with proper citations
- [ ] Future work section is thoughtful and well-reasoned
- [ ] All code is in the repo (except venv and model weights)
- [ ] T30.pdf and T30.zip ready for upload to xSITE Dropbox
