# CoderGym
Curriculum-style ML tasks with specs, tests, and artifacts for training and evaluating code LLMs.

---

## New Tasks — CMPE258 SP26 HW1

Four new `pytorch_task_v1`-compliant tasks added across four new series.
All implemented in pure PyTorch, self-verifiable via `sys.exit(exit_code)`.

**Environment:** Python 3.12, PyTorch 2.2.2, numpy 1.26.4

**Run any task:**
```bash
python MLtasks/tasks/<task_id>/task.py
```

---

### Task 1 — `contrastive_lvl1_simclr_mnist`
**Series:** Contrastive Learning | **Level:** 1

SimCLR self-supervised learning on MNIST. Two random augmented views per image,
NT-Xent contrastive loss on projection head embeddings (no labels during training).
Evaluation via frozen kNN probe (k=5, cosine) on encoder representations.

**Key design choices:**
- Optimizer: Adam, lr=3e-4, cosine LR schedule
- Temperature=0.5; projection head: 128→256→128
- Dataset: 10,000 MNIST images (unlabeled)

**Run output:**
```
============================================================
SimCLR Contrastive Learning on MNIST
============================================================
Device: cpu
Contrastive batches: 39
Model parameters: 159,040
Epoch [1/20]   NT-Xent Loss: 5.3136  LR: 0.000298
Epoch [5/20]   NT-Xent Loss: 4.6102  LR: 0.000256
Epoch [10/20]  NT-Xent Loss: 4.5418  LR: 0.000150
Epoch [15/20]  NT-Xent Loss: 4.5189  LR: 0.000044
Epoch [20/20]  NT-Xent Loss: 4.5132  LR: 0.000000

============================================================
RESULTS
============================================================
  NT-Xent Loss (initial): 5.3136
  NT-Xent Loss (final):   4.5132
  kNN Probe Accuracy:     0.8841
Artifacts saved to output/contrastive_lvl1_simclr_mnist

============================================================
QUALITY CHECKS
============================================================
  PASS  NT-Xent loss decreased: 5.3136 -> 4.5132
  PASS  kNN probe accuracy > 0.85: 0.8841

PASS: All quality checks passed!
============================================================
exit_code: 0
```

| Metric | Value | Threshold | Status |
|---|---|---|---|
| NT-Xent Loss (initial → final) | 5.3136 → 4.5132 | decreasing | PASS |
| kNN Probe Accuracy | **0.8841** | > 0.85 | PASS |

---

### Task 2 — `diffusion_lvl1_ddpm_2d`
**Series:** Diffusion Models | **Level:** 1

DDPM on 2D Swiss Roll data. Linear noise schedule (β₁=1e-4 → β_T=0.02, T=500),
MLP epsilon-predictor with sinusoidal time embeddings, full forward/reverse diffusion chain.
Generates samples and compares mean/std to training distribution.

**Key design choices:**
- Architecture: 4-layer MLP with SiLU activations + sinusoidal time embedding (dim=64)
- Optimizer: Adam, lr=1e-3, cosine LR schedule, grad clip 1.0
- Dataset: 4000 points from sklearn Swiss Roll, projected to 2D and standardized

**Run output:**
```
============================================================
DDPM Diffusion Model on 2D Swiss Roll
============================================================
Device: cpu
Training points: 4000
Data mean: [-1.57e-08  1.19e-08], std: [1.0 1.0]
Model parameters: 149,250
Epoch [1/80]   MSE Loss: 0.794730
Epoch [20/80]  MSE Loss: 0.394045
Epoch [40/80]  MSE Loss: 0.386146
Epoch [60/80]  MSE Loss: 0.395623
Epoch [80/80]  MSE Loss: 0.397234

============================================================
RESULTS
============================================================
  Loss (initial):     0.794730
  Loss (final):       0.397234
  Sample mean error:  0.0207
  Sample std error:   0.0050
Artifacts saved to output/diffusion_lvl1_ddpm_2d

============================================================
QUALITY CHECKS
============================================================
  PASS  Denoising loss decreased: 0.794730 -> 0.397234
  PASS  Sample mean error < 0.5: 0.0207
  PASS  Sample std error < 0.5: 0.0050

PASS: All quality checks passed!
============================================================
exit_code: 0
```

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Denoising MSE (initial → final) | 0.7947 → 0.3972 | decreasing | PASS |
| Generated sample mean error | **0.0207** | < 0.5 | PASS |
| Generated sample std error | **0.0050** | < 0.5 | PASS |

---

### Task 3 — `tcn_lvl1_sequence_cls`
**Series:** Temporal Convolutional Networks | **Level:** 1

TCN with dilated causal convolutions for 4-class synthetic time series classification.
Classes are sinusoids at 1, 2, 4, 8 Hz with Gaussian noise. Dilation doubles per layer
(d=1,2,4,8), residual skip connections, left-pad-only causal convolutions.

**Key design choices:**
- Architecture: input projection → 4 ResidualTCNBlocks (d=1,2,4,8) → GlobalAvgPool → Linear
- 32 filters, kernel size 3; total receptive field = 1 + 2*(2^4 - 1) = 31 timesteps
- Optimizer: Adam, lr=1e-3, cosine LR, grad clip 1.0; 500 samples/class, seq_len=128

**Run output:**
```
============================================================
TCN Sequence Classification on Synthetic Time Series
============================================================
Device: cpu
Train samples: 1600 | Val samples: 400 | Classes: 4
Model parameters: 10,532
Epoch [1/50]   Train Loss: 0.8819  Val Loss: 0.5439  Val Acc: 1.0000
Epoch [10/50]  Train Loss: 0.0076  Val Loss: 0.0035  Val Acc: 1.0000
Epoch [20/50]  Train Loss: 0.0031  Val Loss: 0.0008  Val Acc: 1.0000
Epoch [30/50]  Train Loss: 0.0016  Val Loss: 0.0005  Val Acc: 1.0000
Epoch [40/50]  Train Loss: 0.0014  Val Loss: 0.0004  Val Acc: 1.0000
Epoch [50/50]  Train Loss: 0.0014  Val Loss: 0.0003  Val Acc: 1.0000

RESULTS
  Train Accuracy: 1.0000 | Val Accuracy: 1.0000
  Train Loss:     0.0003 | Val Loss:     0.0003

Classification Report (val):
              precision  recall  f1-score  support
     class_0     1.00    1.00     1.00      111
     class_1     1.00    1.00     1.00       99
     class_2     1.00    1.00     1.00       96
     class_3     1.00    1.00     1.00       94
    accuracy                      1.00      400

============================================================
QUALITY CHECKS
============================================================
  PASS  Val accuracy > 0.85: 1.0000
  PASS  Train loss decreased: 0.8819 -> 0.0014
  PASS  Val loss < 0.5: 0.0003

PASS: All quality checks passed!
============================================================
exit_code: 0
```

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Val Accuracy | **1.0000** | > 0.85 | PASS |
| Train Loss (initial → final) | 0.8819 → 0.0014 | decreasing | PASS |
| Val Loss | **0.0003** | < 0.5 | PASS |

---

### Task 4 — `distillation_lvl1_kd_mnist`
**Series:** Knowledge Distillation | **Level:** 1

Teacher-student knowledge distillation on MNIST. Large TeacherCNN trained first with
standard cross-entropy. StudentCNN (187× fewer parameters) then trained with combined
soft-target KL divergence loss + hard-label CE loss at temperature T=4.

**Key design choices:**
- Teacher: 3 conv blocks + BN + dropout + FC head (1,701,578 params)
- Student: 2 conv blocks + FC head (9,098 params) — 187× compression
- Loss: L = (1−α)·CE(y, p_S) + α·T²·KL(p_T∥p_S), α=0.7, T=4.0
- Optimizer: Adam, lr=1e-3, StepLR(step=7, γ=0.5); 15 epochs each

**Run output:**
```
============================================================
Knowledge Distillation on MNIST (Teacher -> Student)
============================================================
Device: cpu
Train samples: 60000 | Val samples: 10000

Teacher params: 1,701,578
Student params:     9,098
Param ratio:       187.0x

--- Training Teacher ---
Epoch [1/15]   [CE]           Loss: 0.1667
Epoch [5/15]   [CE]           Loss: 0.0312
Epoch [10/15]  [CE]           Loss: 0.0083
Epoch [15/15]  [CE]           Loss: 0.0063

Teacher Val Accuracy: 0.9946

--- Training Student with Distillation ---
Epoch [1/15]   [distillation] Loss: 0.4521
Epoch [5/15]   [distillation] Loss: 0.1839
Epoch [10/15]  [distillation] Loss: 0.1374
Epoch [15/15]  [distillation] Loss: 0.1245

Student Val Accuracy: 0.9887

============================================================
RESULTS
============================================================
  Teacher Accuracy: 0.9946
  Student Accuracy: 0.9887
  Accuracy gap:     0.0059
  Param ratio:      187.0x
Artifacts saved to output/distillation_lvl1_kd_mnist

============================================================
QUALITY CHECKS
============================================================
  PASS  Teacher accuracy > 0.99: 0.9946
  PASS  Student accuracy > 0.95: 0.9887
  PASS  Accuracy gap < 0.03: 0.0059
  PASS  Param ratio > 5: 187.0x

PASS: All quality checks passed!
============================================================
exit_code: 0
```

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Teacher Accuracy | **0.9946** | > 0.99 | PASS |
| Student Accuracy | **0.9887** | > 0.95 | PASS |
| Accuracy gap | **0.0059** | < 0.03 | PASS |
| Param ratio (teacher/student) | **187×** | > 5× | PASS |

---

## Summary

| Task | Series | Dataset | Exit Code | All Checks |
|---|---|---|---|---|
| `contrastive_lvl1_simclr_mnist` | Contrastive Learning | MNIST (self-supervised) | 0 | PASS |
| `diffusion_lvl1_ddpm_2d` | Diffusion Models | Synthetic 2D Swiss Roll | 0 | PASS |
| `tcn_lvl1_sequence_cls` | Temporal Convolutional Networks | Synthetic time series | 0 | PASS |
| `distillation_lvl1_kd_mnist` | Knowledge Distillation | MNIST | 0 | PASS |
