# MUESUM — Reproducibility & Verification Guide
**Project**: MUESUM (AI-Powered Historical Artwork Restoration)  
**Standard**: Independent Academic & Production Audit Standard  

---

## 1. Cryptographic Checkpoints Verification

Verify checkpoint integrity using standard SHA-256 hashes:

| Checkpoint Name | Filepath | Size (Bytes) | SHA-256 Checksum |
|---|---|:---:|:---:|
| **Baseline Model (Epoch 8)** | `checkpoints/baseline_run/best_model.pth` | 339,079,693 | `0FECFFBF0D43BC03760A34E041A69C13FCDE00E3A7C914D3A3D824F4873AF8B7` |
| **Experimental Texture Model**| `checkpoints/experimental_texture_model/best_model.pth` | 339,080,013 | `14481951B79F4DF6FEFC0A115677DB4E6D2B9564A2A9294A96162912AE5A9A9F` |

To verify on Windows PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 checkpoints\baseline_run\best_model.pth
Get-FileHash -Algorithm SHA256 checkpoints\experimental_texture_model\best_model.pth
```

---

## 2. Environment Setup

### 2.1 Dependencies Installation
```bash
git clone https://github.com/your-username/MUESUM.git
cd MUESUM
pip install -r requirements.txt
```

### 2.2 Determinism Guarantees
All training, damage simulation, and evaluation scripts enforce deterministic seeding:
```python
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## 3. Step-by-Step Reproduction Pipeline

### Step 1: Independent Holdout Dataset Acquisition
Acquire the 40 independent public-domain artworks from The Met Collection API v1.1 and generate synthetic damage and masks:
```bash
python scripts/acquire_independent_holdout.py
```
*Outputs: `data/independent_holdout/clean/`, `damaged/`, `masks/`, and `metadata.csv`.*

### Step 2: Experimental Model Fine-Tuning
Execute fine-tuning with Multi-Term Composite Loss (Masked L1 + Sobel Edge + TV + VGG Perceptual):
```bash
python scripts/train_experimental_model.py
```
*Outputs: `checkpoints/experimental_texture_model/best_model.pth` (Val SSIM: 0.5544).*

### Step 3: Final Independent Evaluation
Benchmark both the Baseline Model and Experimental Texture Model on the 40 holdout artworks:
```bash
python scripts/evaluate_independent_holdout.py
```
*Outputs: `reports/final_independent_evaluation/holdout_metrics.csv` and visual comparison figures.*

### Step 4: Execute Full Automated Test Suite
Run all 23 automated unit and regression tests:
```bash
python tests/run_all_tests.py
```
*Expected Result: `Ran 23 tests ... OK (100% Passed)`.*

### Step 5: Launch Gradio Restoration Studio
Launch the interactive web application:
```bash
python app/demo.py
# Or on Windows:
run_muesum.bat
```
Visit `http://localhost:7860` to access the interactive studio.
