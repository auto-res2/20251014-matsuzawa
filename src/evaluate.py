import json
import os
from pathlib import Path
from typing import List, Dict

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import pandas as pd


@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    results_dir = Path(cfg.results_dir)
    run_ids: List[str] = cfg.all_runs

    records: List[Dict] = []
    for rid in run_ids:
        res_file = results_dir / rid / "results.json"
        if not res_file.exists():
            print(f"Warning: missing results for {rid}")
            continue
        records.append(json.loads(res_file.read_text()))

    if not records:
        print("No result files found – aborting evaluation.")
        return

    df = pd.DataFrame(records)
    best = df.loc[df["best_val_acc"].idxmax()]
    summary = {
        "best_run": best["run_id"],
        "best_val_acc": best["best_val_acc"],
        "mean_val_acc": df["best_val_acc"].mean(),
        "std_val_acc": df["best_val_acc"].std(),
    }

    # Plot comparison
    plt.figure(figsize=(8, 4))
    plt.bar(df["run_id"], df["best_val_acc"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation Accuracy")
    plt.title("Best validation accuracy across runs")
    plt.tight_layout()

    fig_path = results_dir / "comparison.png"
    plt.savefig(fig_path)

    # Log to WandB if available
    if os.getenv("WANDB_API_KEY"):
        import wandb
        with wandb.init(project="251014-test", entity="gengaru617", job_type="evaluation") as wb:
            wb.log(summary)
            wb.log({"comparison": wandb.Image(str(fig_path))})
            wb.save(str(fig_path))
            print(f"Logged evaluation results to WandB: {wb.url}")

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
