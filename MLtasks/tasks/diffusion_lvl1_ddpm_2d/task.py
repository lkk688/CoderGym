"""
DDPM Denoising Diffusion Probabilistic Model on 2D Swiss Roll

Mathematical Formulation:
- Forward process: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I)
- Closed form: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
- Reverse process: p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2*I)
- Training objective: L = E[||eps - eps_theta(x_t, t)||^2]
- alpha_bar_t = prod_{s=1}^{t}(1 - beta_s)  (linear noise schedule)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_swiss_roll
from typing import Dict, Any, Tuple

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_name": "diffusion_lvl1_ddpm_2d",
        "series": "Diffusion Models",
        "level": 1,
        "description": "DDPM on 2D Swiss Roll: linear noise schedule, epsilon-prediction network, reverse sampling",
        "input_type": "float2d",
        "output_type": "float2d",
        "metrics": ["train_loss", "sample_mean_error", "sample_std_error"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    n_samples: int = 4000,
    batch_size: int = 256,
) -> Tuple[DataLoader, np.ndarray]:
    """
    Returns:
        loader: DataLoader of normalised 2D Swiss Roll points
        data_np: raw numpy array (for stats comparison)
    """
    X, _ = make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42)
    data = X[:, [0, 2]].astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    dataset = TensorDataset(torch.FloatTensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, data


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (np.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredictor(nn.Module):
    """Small MLP that predicts noise eps given (x_t, t)."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


def build_model(data_dim: int = 2, device: torch.device = None) -> NoisePredictor:
    if device is None:
        device = get_device()
    return NoisePredictor(data_dim=data_dim).to(device)


class DDPMScheduler:
    """Linear beta schedule and diffusion utilities."""

    def __init__(self, T: int = 500, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        self.T = T
        self.device = device or torch.device("cpu")
        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()
        self.posterior_variance = betas * (1.0 - torch.cat([alpha_bar[:1], alpha_bar[:-1]])) / (1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t|x_0)."""
        eps = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_om_ab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_om_ab * eps, eps

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t_idx: int) -> torch.Tensor:
        """One reverse step: sample x_{t-1} from p_theta(x_{t-1}|x_t)."""
        t_tensor = torch.full((x.size(0),), t_idx, device=self.device, dtype=torch.long)
        eps_pred = model(x, t_tensor)

        coeff = self.betas[t_idx] / self.sqrt_one_minus_alpha_bar[t_idx]
        mean = self.sqrt_recip_alphas[t_idx] * (x - coeff * eps_pred)

        if t_idx == 0:
            return mean
        noise = torch.randn_like(x)
        return mean + self.posterior_variance[t_idx].sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, data_dim: int = 2) -> torch.Tensor:
        """Full reverse chain from x_T ~ N(0,I) to x_0."""
        x = torch.randn(n, data_dim, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    T: int = 500,
) -> Dict[str, Any]:
    scheduler = DDPMScheduler(T=T, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (x0,) in loader:
            x0 = x0.to(device)
            t = torch.randint(0, T, (x0.size(0),), device=device)
            x_t, eps = scheduler.q_sample(x0, t)
            eps_pred = model(x_t, t)
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        lr_sched.step()
        avg = epoch_loss / n_batches
        history["loss"].append(avg)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  MSE Loss: {avg:.6f}")

    return {"history": history, "final_loss": history["loss"][-1], "scheduler": scheduler}


def evaluate(
    model: nn.Module,
    data_np: np.ndarray,
    train_result: Dict[str, Any],
    device: torch.device,
    n_samples: int = 500,
) -> Dict[str, Any]:
    """Generate samples and compare stats to training data."""
    sched: DDPMScheduler = train_result["scheduler"]
    samples = sched.sample(model, n=n_samples, data_dim=2).cpu().numpy()

    data_mean = data_np.mean(axis=0)
    data_std = data_np.std(axis=0)
    samp_mean = samples.mean(axis=0)
    samp_std = samples.std(axis=0)

    mean_err = float(np.abs(samp_mean - data_mean).mean())
    std_err = float(np.abs(samp_std - data_std).mean())

    history = train_result["history"]
    return {
        "final_loss": history["loss"][-1],
        "initial_loss": history["loss"][0],
        "loss_decreased": bool(history["loss"][-1] < history["loss"][0]),
        "sample_mean_error": mean_err,
        "sample_std_error": std_err,
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict noise for noisy inputs x at timestep 0 (for API compliance)."""
    model.eval()
    x_t = torch.FloatTensor(x).to(device)
    t = torch.zeros(x_t.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        eps = model(x_t, t)
    return eps.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    metrics: Dict[str, Any],
    save_dir: str = "output",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "noise_predictor.pth"))
    metrics_to_save = {k: v for k, v in metrics.items() if not isinstance(v, DDPMScheduler)}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Artifacts saved to {save_dir}")


def main() -> int:
    print("=" * 60)
    print("DDPM Diffusion Model on 2D Swiss Roll")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    loader, data_np = make_dataloaders(n_samples=4000, batch_size=256)
    print(f"Training points: {len(loader.dataset)}")
    print(f"Data mean: {data_np.mean(axis=0)}, std: {data_np.std(axis=0)}")

    model = build_model(data_dim=2, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_result = train(model, loader, device, epochs=80, lr=1e-3, T=500)

    metrics = evaluate(model, data_np, train_result, device, n_samples=500)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Loss (initial):     {metrics['initial_loss']:.6f}")
    print(f"  Loss (final):       {metrics['final_loss']:.6f}")
    print(f"  Sample mean error:  {metrics['sample_mean_error']:.4f}")
    print(f"  Sample std error:   {metrics['sample_std_error']:.4f}")

    save_dir = "output/diffusion_lvl1_ddpm_2d"
    all_metrics = {**metrics, "loss_history": train_result["history"]["loss"]}
    save_artifacts(model, all_metrics, save_dir)

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    check1 = metrics["loss_decreased"]
    print(f"  {'PASS' if check1 else 'FAIL'} Denoising loss decreased: "
          f"{metrics['initial_loss']:.6f} -> {metrics['final_loss']:.6f}")
    checks_passed = checks_passed and check1

    check2 = metrics["sample_mean_error"] < 0.5
    print(f"  {'PASS' if check2 else 'FAIL'} Sample mean error < 0.5: {metrics['sample_mean_error']:.4f}")
    checks_passed = checks_passed and check2

    check3 = metrics["sample_std_error"] < 0.5
    print(f"  {'PASS' if check3 else 'FAIL'} Sample std error < 0.5: {metrics['sample_std_error']:.4f}")
    checks_passed = checks_passed and check3

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
