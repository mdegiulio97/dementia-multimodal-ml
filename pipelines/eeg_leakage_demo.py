"""How much does a leaky split inflate EEG performance? (ds004504, open data)

    uv run python pipelines/eeg_leakage_demo.py

Same features and models as pipelines/eeg.py; the only difference is the cross-validation:
epoch-level folds (epochs of one person in both train and test, metrics over epochs) versus
the package's participant-level folds (metrics over participants).
"""

import json

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from dementia_ml import paths, provenance
from dementia_ml.eeg import classify
from dementia_ml.eeg.run import cache_path, extract
from dementia_ml.eval import bootstrap, metrics


def leaky(features, labels, task, model, seed=0, n_splits=5):
    classes, positive = classify.TASKS[task]
    keep = labels.isin(classes).to_numpy()
    X, y = features[keep].reset_index(drop=True), (labels[keep] == positive).astype(int).to_numpy()
    prob = np.zeros(len(X))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        clf = classify._model(model, seed).fit(X.iloc[tr], y[tr])
        prob[te] = clf.predict_proba(X.iloc[te])[:, 1]
    est, lo, hi = bootstrap.ci(
        lambda i: metrics.auc(y[i], prob[i]), n=len(y), n_boot=200, seed=seed
    )
    bacc = metrics.balanced_accuracy(y, (prob >= 0.5).astype(int))
    return {
        "auc": {"estimate": est, "ci_low": lo, "ci_high": hi},
        "balanced_accuracy": bacc,
        "n_epochs": len(y),
    }


def main() -> None:
    with open("configs/eeg.yaml") as fh:
        cfg = yaml.safe_load(fh)
    feats = extract(
        paths.data_root() / cfg["bids_dir"], cfg, cache_path(paths.data_root() / "eeg", cfg)
    )
    X = feats.drop(columns=["subject", "group"])
    proper = json.loads((paths.results_dir("open") / "eeg_metrics.json").read_text())["results"]
    out = {}
    for task in ["AD_vs_CN", "FTD_vs_CN", "AD_vs_FTD"]:
        for model in ["logreg", "lgbm"]:
            lk = leaky(X, feats["group"], task, model)
            pr = proper[task][model]
            out[f"{task}/{model}"] = {
                "leaky_epoch_level": lk,
                "participant_level": {
                    "auc": pr["auc"],
                    "balanced_accuracy": pr["balanced_accuracy"]["estimate"],
                },
            }
            print(
                f"{task:10s} {model:6s} leaky AUC {lk['auc']['estimate']:.3f} BA {lk['balanced_accuracy']:.3f} | "
                f"participant AUC {pr['auc']['estimate']:.3f} BA {pr['balanced_accuracy']['estimate']:.3f}",
                flush=True,
            )
    payload = {
        "question": "Inflation of EEG classification performance when epochs of one participant cross folds",
        "results": out,
        "provenance": provenance.block("OpenNeuro ds004504", "1.0.9", cfg, 0),
    }
    (paths.results_dir("open") / "eeg_leakage_demo.json").write_text(json.dumps(payload, indent=2))
    figure(out, paths.figures_dir("open") / "eeg_leakage_demo.png")


def figure(out: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(out)
    labels = [k.replace("_vs_", " vs ").replace("/", chr(10)) for k in keys]
    leaky_auc = [out[k]["leaky_epoch_level"]["auc"]["estimate"] for k in keys]
    proper_auc = [out[k]["participant_level"]["auc"]["estimate"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, leaky_auc, 0.4, color="#e76f51", label="epoch-level folds (leaky)")
    ax.bar(x + 0.2, proper_auc, 0.4, color="#264653", label="participant-level folds")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
