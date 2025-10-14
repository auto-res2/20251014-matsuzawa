import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports – Hydra automatically adds the project root to PYTHONPATH
from src.preprocess import get_datasets
from src.model import build_model

# -----------------------------
# Helper utilities
# -----------------------------

def _init_wandb(cfg: DictConfig) -> Any:
    """Initialise Weights & Biases run and persist metadata."""
    import wandb  # heavy import – do it lazily

    entity = cfg.get("wandb", {}).get("entity", "gengaru617")
    project = cfg.get("wandb", {}).get("project", "251014-test")
    run_name = cfg.get("wandb", {}).get("run_name", cfg.run_id)
    tags = cfg.get("wandb", {}).get("tags", [cfg.method, cfg.model.name])

    wandb_run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        reinit=True,
    )

    # Persist metadata for CI
    iteration = os.getenv("EXPERIMENT_ITERATION", "0")
    meta_dir = Path(f".research/iteration{iteration}")
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = meta_dir / "wandb_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "wandb_entity": entity,
                "wandb_project": project,
                "wandb_run_id": wandb_run.id,
            },
            indent=2,
        )
    )
    print(f"WandB run initialised at {wandb_run.url}")
    return wandb_run


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module):
    """Factory for optimizer and scheduler."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt_type = cfg.training.optimizer.lower()
    if opt_type == "sgd":
        optimizer = optim.SGD(
            params,
            lr=cfg.training.learning_rate,
            momentum=cfg.training.get("momentum", 0.0),
            weight_decay=cfg.training.get("weight_decay", 0.0),
        )
    elif opt_type in ("adam", "adamw"):
        adam_class = optim.AdamW if opt_type == "adamw" else optim.Adam
        optimizer = adam_class(
            params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer {opt_type}")

    # Scheduler
    sched_cfg = cfg.training.get("lr_scheduler", None)
    scheduler = None
    if sched_cfg:
        sched_type = sched_cfg.type.lower()
        if sched_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sched_cfg.T_max
            )
        elif sched_type == "linear":
            warmup_steps = int(sched_cfg.get("warmup_steps", 0))
            # Auto-compute total_steps when not provided
            total_steps = sched_cfg.get("total_steps", None)
            if total_steps is None:
                # total_steps = epochs * iterations_per_epoch
                # We defer the actual value computation; use Lambda LR updating per step
                total_steps = 0  # placeholder updated later
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 1.0 - progress)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            raise ValueError(f"Unsupported scheduler type {sched_type}")
    return optimizer, scheduler


def _accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        return (preds == target).float().mean()


def _save_results(run_dir: Path, results: Dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))


# -----------------------------
# Training loop
# -----------------------------

def run_training(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_set, val_set, num_classes = get_datasets(cfg)
    g_accum = cfg.training.get("gradient_accumulation_steps", 1)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = build_model(cfg, num_classes=num_classes).to(device)

    # Criterion
    if cfg.dataset.name.lower().startswith("cifar"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optim / sched
    optimizer, scheduler = _build_optimizer(cfg, model)

    # Fix total_steps for linear scheduler if needed
    if scheduler and isinstance(scheduler, optim.lr_scheduler.LambdaLR):
        total_steps = cfg.training.epochs * math.ceil(len(train_loader.dataset) / cfg.training.batch_size / g_accum)
        scheduler.lr_lambdas[0].__defaults__ = (  # type: ignore
            scheduler.lr_lambdas[0].__defaults__[0],  # warmup_steps remains
            total_steps,
        )

    # WandB
    wb = _init_wandb(cfg)

    global_step = 0
    best_val_acc = 0.0
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y) / g_accum
            loss.backward()

            if i % g_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                global_step += 1

            acc = _accuracy(outputs, y).item()
            epoch_loss += loss.item() * g_accum
            epoch_acc += acc
            if i % 10 == 0:
                wb.log({"train/loss": loss.item() * g_accum, "train/acc": acc, "global_step": global_step})

        epoch_loss /= len(train_loader)
        epoch_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_acc += _accuracy(outputs, y).item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        best_val_acc = max(best_val_acc, val_acc)

        wb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": epoch_loss,
                "train/epoch_acc": epoch_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # Checkpoint – light to save memory
        ckpt_path = Path(cfg.results_dir) / cfg.run_id / f"epoch{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_path)
        wb.save(str(ckpt_path))

    wb.finish()

    results = {
        "run_id": cfg.run_id,
        "best_val_acc": best_val_acc,
        "history": history,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _save_results(Path(cfg.results_dir) / cfg.run_id, results)
    print(json.dumps(results))


# -----------------------------
# Hydra main
# -----------------------------

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # The working dir is changed by Hydra: recover original path for outputs
    orig_cwd = Path(hydra.utils.get_original_cwd())
    cfg.results_dir = os.path.abspath(cfg.get("results_dir", os.path.join(orig_cwd, "results")))
    run_training(cfg)


if __name__ == "__main__":
    main()
