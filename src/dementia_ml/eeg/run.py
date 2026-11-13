"""ds004504: spectral features and subject-level classification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from dementia_ml import paths, provenance  # noqa: E402
from dementia_ml.eeg import classify, features, io  # noqa: E402
from dementia_ml.mr.sources import config as source_config  # noqa: E402

COLORS = {"CN": "#2a9d8f", "AD": "#e76f51", "FTD": "#264653"}
GROUPS = ("CN", "AD", "FTD")


CACHE_KEYS = ("epoch_s", "overlap", "use_derivatives", "fmin")


def cache_path(root: Path, cfg: dict) -> Path:
    """Feature cache named after every parameter that changes the features."""
    key = {k: cfg.get(k, features.FMIN if k == "fmin" else None) for k in CACHE_KEYS}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return Path(root) / f"features_ds004504_{digest}.parquet"


def extract(bids_root: Path, cfg: dict, cache: Path) -> pd.DataFrame:
    """Epoch features for every participant (cached as Parquet next to the data)."""
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for row in io.participants(bids_root).itertuples():
        raw = io.load_subject(bids_root, row.subject, cfg["use_derivatives"])
        f = features.epoch_features(raw, cfg["epoch_s"], cfg["overlap"])
        parts.append(f.assign(subject=row.subject, group=row.group))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cache)
    return out


def _band_figure(feats: pd.DataFrame, path: Path) -> None:
    per_subject = feats.drop(columns="group").groupby("subject").mean(numeric_only=True)
    group = feats.groupby("subject")["group"].first()
    fig, axes = plt.subplots(1, len(features.BANDS), figsize=(12, 3.2))
    for ax, band in zip(axes, features.BANDS, strict=True):
        cols = [c for c in per_subject.columns if c.endswith(f"_rel_{band}")]
        val = per_subject[cols].mean(axis=1)
        bp = ax.boxplot(
            [val[group == g] for g in GROUPS],
            tick_labels=list(GROUPS),
            patch_artist=True,
            widths=0.6,
        )
        for patch, g in zip(bp["boxes"], GROUPS, strict=True):
            patch.set_facecolor(COLORS[g])
            patch.set_alpha(0.6)
        lo, hi = features.BANDS[band]
        ax.set_title(f"{band} ({lo:g}-{hi:g} Hz)", fontsize=9)
    axes[0].set_ylabel("Relative power (mean of 19 channels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roc_figure(res: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for model, r in res.items():
        p = r["subject_predictions"]
        y = (p["label"] == r["classes"][1]).astype(int)
        fpr, tpr, _ = roc_curve(y, p["prob"])
        a = r["auc"]
        label = f"{model}: AUC {a['estimate']:.2f} ({a['ci_low']:.2f}-{a['ci_high']:.2f})"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    first = next(iter(res.values()))
    ax.set_title(f"{first['classes'][1]} vs {first['classes'][0]} (per participant)", fontsize=10)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _topomap_figure(feats: pd.DataFrame, path: Path) -> None:
    channels = source_config("eeg_ds004504")["channels"]
    info = mne.create_info(channels, 500.0, "eeg")
    info.set_montage("standard_1020")
    n_per_group = feats.groupby("subject")["group"].first().value_counts()
    values = {}
    for g in GROUPS:
        sub = feats[feats["group"] == g].groupby("subject").mean(numeric_only=True)
        values[g] = np.array([np.log10(sub[f"{ch}_theta_alpha"]).mean() for ch in channels])
    vmin = min(v.min() for v in values.values())
    vmax = max(v.max() for v in values.values())
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    im = None
    for ax, g in zip(axes, GROUPS, strict=True):
        im, _ = mne.viz.plot_topomap(
            values[g], info, axes=ax, show=False, vlim=(vmin, vmax), cmap="RdBu_r"
        )
        ax.set_title(f"{g} (n={n_per_group.get(g, 0)})")
    fig.colorbar(im, ax=axes, shrink=0.7, label="log10 theta/alpha power ratio")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: dict, out_root: Path | None = None) -> dict:
    out_root = out_root or paths.REPO_ROOT
    bids_root = paths.data_root() / cfg["bids_dir"]
    feats = extract(bids_root, cfg, cache_path(paths.data_root() / "eeg", cfg))
    X = feats.drop(columns=["subject", "group"])
    res_dir, fig_dir = paths.results_dir("open", out_root), paths.figures_dir("open", out_root)
    results, preds = {}, []
    for task in cfg["tasks"]:
        results[task] = {}
        for model in cfg["models"]:
            r = classify.run(
                X,
                feats["group"],
                feats["subject"],
                task=task,
                model=model,
                seed=cfg["seed"],
                n_repeats=cfg["n_repeats"],
                n_boot=cfg["n_boot"],
            )
            results[task][model] = r
            preds.append(r["subject_predictions"].assign(task=task, model=model))
        if task != "3class":
            _roc_figure(results[task], fig_dir / f"eeg_roc_{task}.png")
    _band_figure(feats, fig_dir / "eeg_band_power_by_group.png")
    _topomap_figure(feats, fig_dir / "eeg_topomap_theta_alpha.png")
    pd.concat(preds).rename_axis("subject").to_csv(res_dir / "eeg_subject_predictions.csv")
    summary = {
        "question": "Does resting-state EEG separate AD, FTD and controls at the participant level?",
        "dataset": "OpenNeuro ds004504",
        "n_subjects": int(feats["subject"].nunique()),
        "n_epochs": len(feats),
        "results": {
            t: {m: {k: v for k, v in r.items() if k != "subject_predictions"} for m, r in d.items()}
            for t, d in results.items()
        },
        "provenance": provenance.block(
            "OpenNeuro ds004504", source_config("eeg_ds004504")["doi"], cfg, cfg["seed"]
        ),
    }
    (res_dir / "eeg_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
