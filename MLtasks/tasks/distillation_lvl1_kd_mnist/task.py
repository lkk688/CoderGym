"""
Knowledge Distillation: Teacher -> Student on MNIST

Mathematical Formulation:
- Soft target distribution: p_T(x; T) = softmax(z_teacher / T)
- Distillation loss: L_KD = T^2 * KL(p_T(x;T) || p_S(x;T))
  where KL(P||Q) = sum_i P_i * log(P_i / Q_i)
- Total student loss: L = (1 - alpha)*CE(y_hard, p_S) + alpha*L_KD
- alpha in [0,1] balances hard labels vs soft targets; T > 1 softens logits.

Teacher: 4-layer CNN (~200k params)
Student: 2-layer tiny CNN (~20k params, <1/5 of teacher)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "distillation_lvl1_kd_mnist",
        "series": "Knowledge Distillation",
        "level": 1,
        "description": "Teacher-student knowledge distillation with soft targets and temperature scaling on MNIST",
        "input_type": "image",
        "output_type": "class_label",
        "metrics": ["teacher_accuracy", "student_accuracy", "param_ratio"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


class TeacherCNN(nn.Module):
    """Larger teacher CNN (~200k parameters)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class StudentCNN(nn.Module):
    """Compact student CNN (~20k parameters)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(role: str = "teacher", device: torch.device = None) -> nn.Module:
    """
    Args:
        role: 'teacher' or 'student'
    """
    if device is None:
        device = get_device()
    if role == "teacher":
        return TeacherCNN().to(device)
    return StudentCNN().to(device)


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return (1.0 - alpha) * ce_loss + alpha * kd_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    teacher: nn.Module = None,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """
    Train either a teacher (teacher=None, standard CE) or a student (teacher provided).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    if teacher is not None:
        teacher.eval()

    history = {"loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss = _distillation_loss(logits, teacher_logits, y, temperature, alpha)
            else:
                loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = total_loss / n_batches
        history["loss"].append(avg)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            mode = "distillation" if teacher is not None else "CE"
            print(f"Epoch [{epoch+1}/{epochs}]  [{mode}]  Loss: {avg:.4f}")

    return {"history": history}


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y).item()
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    return {
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "loss": total_loss / len(data_loader),
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_t = torch.FloatTensor(x).to(device)
    with torch.no_grad():
        logits = model(x_t)
    return logits.argmax(dim=1).cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_dir: str = "output",
    name: str = "model",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{name}.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {save_dir}")


def main() -> int:
    print("=" * 60)
    print("Knowledge Distillation on MNIST (Teacher -> Student)")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders(batch_size=128)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    teacher = build_model("teacher", device)
    student = build_model("student", device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    param_ratio = teacher_params / student_params
    print(f"\nTeacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}")
    print(f"Param ratio (teacher/student): {param_ratio:.1f}x")

    print("\n--- Training Teacher ---")
    train(teacher, train_loader, device, epochs=15, lr=1e-3, teacher=None)

    print("\n--- Evaluating Teacher ---")
    teacher_val = evaluate(teacher, val_loader, device)
    print(f"Teacher Val Accuracy: {teacher_val['accuracy']:.4f}")

    print("\n--- Training Student with Distillation ---")
    train(student, train_loader, device, epochs=15, lr=1e-3,
          teacher=teacher, temperature=4.0, alpha=0.7)

    print("\n--- Evaluating Student ---")
    student_val = evaluate(student, val_loader, device)
    print(f"Student Val Accuracy: {student_val['accuracy']:.4f}")

    save_dir = "output/distillation_lvl1_kd_mnist"
    all_metrics = {
        "teacher_accuracy": teacher_val["accuracy"],
        "student_accuracy": student_val["accuracy"],
        "teacher_loss": teacher_val["loss"],
        "student_loss": student_val["loss"],
        "teacher_params": teacher_params,
        "student_params": student_params,
        "param_ratio": param_ratio,
    }
    save_artifacts(student, all_metrics, save_dir, name="student")
    torch.save(teacher.state_dict(), os.path.join(save_dir, "teacher.pth"))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Teacher Accuracy: {teacher_val['accuracy']:.4f}")
    print(f"  Student Accuracy: {student_val['accuracy']:.4f}")
    print(f"  Accuracy gap:     {abs(teacher_val['accuracy'] - student_val['accuracy']):.4f}")
    print(f"  Param ratio:      {param_ratio:.1f}x")

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    check1 = teacher_val["accuracy"] > 0.99
    print(f"  {'PASS' if check1 else 'FAIL'} Teacher accuracy > 0.99: {teacher_val['accuracy']:.4f}")
    checks_passed = checks_passed and check1

    check2 = student_val["accuracy"] > 0.95
    print(f"  {'PASS' if check2 else 'FAIL'} Student accuracy > 0.95: {student_val['accuracy']:.4f}")
    checks_passed = checks_passed and check2

    gap = abs(teacher_val["accuracy"] - student_val["accuracy"])
    check3 = gap < 0.03
    print(f"  {'PASS' if check3 else 'FAIL'} Accuracy gap < 0.03: {gap:.4f}")
    checks_passed = checks_passed and check3

    check4 = param_ratio > 5.0
    print(f"  {'PASS' if check4 else 'FAIL'} Param ratio > 5: {param_ratio:.1f}x")
    checks_passed = checks_passed and check4

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
