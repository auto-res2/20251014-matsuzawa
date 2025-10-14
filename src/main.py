import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


# --------------------------------------------------
# Helper for tee-like logging
# --------------------------------------------------


def _run_subprocess(cmd: List[str], run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    with stdout_path.open("w") as out_f, stderr_path.open("w") as err_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
        )
        # Stream output live while recording
        for line in process.stdout:  # type: ignore
            sys.stdout.write(line)
            out_f.write(line)
            out_f.flush()
        for line in process.stderr:  # type: ignore
            sys.stderr.write(line)
            err_f.write(line)
            err_f.flush()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    orig_cwd = Path(hydra.utils.get_original_cwd())
    results_dir = Path(cfg.get("results_dir", orig_cwd / "results")).absolute()

    run_ids: List[str]
    if cfg.run == "all":
        run_ids = cfg.all_runs
    else:
        run_ids = [cfg.run]

    for run_id in run_ids:
        print("=" * 80)
        print(f"Launching experiment {run_id}")
        print("=" * 80)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            f"run={run_id}",
            f"results_dir={str(results_dir)}",
        ]
        _run_subprocess(cmd, results_dir / run_id)

    # After all experiments, run evaluation
    print("All experiments finished. Launching evaluation...")
    eval_cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluate",
        f"results_dir={str(results_dir)}",
    ]
    _run_subprocess(eval_cmd, results_dir / "evaluation")


if __name__ == "__main__":
    main()
