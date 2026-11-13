"""EEGNet on raw resting-state windows of ds004504 (CPU), same subject-level folds.

uv run --extra dl python pipelines/eeg_eegnet.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from dementia_ml import paths, provenance
from dementia_ml.eeg import deep
from dementia_ml.eeg.run import _roc_figure
from dementia_ml.mr.sources import config as source_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/eeg_eegnet.yaml"))
    cfg = yaml.safe_load(ap.parse_args().config.read_text())
    torch.set_num_threads(cfg.get("threads", 8))
    bids = paths.data_root() / cfg["bids_dir"]
    cache = paths.data_root() / "eeg" / f"windows_{cfg['sfreq']:g}hz_{cfg['window_s']:g}s.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, labels, subjects = z["X"], pd.Series(z["labels"]), pd.Series(z["subjects"])
    else:
        X, labels, subjects = deep.make_windows(bids, cfg["sfreq"], cfg["window_s"])
        np.savez(cache, X=X, labels=labels.to_numpy(), subjects=subjects.to_numpy())
    res_dir, fig_dir = paths.results_dir("open"), paths.figures_dir("open")
    results, preds = {}, []
    for task in cfg["tasks"]:
        r = deep.run(
            X,
            labels,
            subjects,
            task,
            seed=cfg["seed"],
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            n_splits=cfg["n_splits"],
            sfreq=cfg["sfreq"],
            n_boot=cfg["n_boot"],
        )
        results[task] = r
        preds.append(r["subject_predictions"].assign(task=task, model="eegnet"))
        _roc_figure({"eegnet": r}, fig_dir / f"eeg_roc_{task}_eegnet.png")
        a = r["auc"]
        print(
            f"{task} eegnet AUC {a['estimate']:.3f} ({a['ci_low']:.3f}-{a['ci_high']:.3f})",
            flush=True,
        )
    pd.concat(preds).rename_axis("subject").to_csv(res_dir / "eeg_subject_predictions_eegnet.csv")
    summary = {
        "question": "Does a convolutional network on raw EEG windows beat spectral features?",
        "dataset": "OpenNeuro ds004504",
        "results": {
            t: {k: v for k, v in r.items() if k != "subject_predictions"}
            for t, r in results.items()
        },
        "provenance": provenance.block(
            "OpenNeuro ds004504", source_config("eeg_ds004504")["doi"], cfg, cfg["seed"]
        ),
    }
    (res_dir / "eeg_metrics_eegnet.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
