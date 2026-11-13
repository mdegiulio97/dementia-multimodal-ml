"""EEGNet on raw resting-state windows, with the same subject-level folds as the
spectral models.

Each outer fold splits participants into train and test; inside the training
participants a further 20% (by participant) is held out for early stopping. A
participant's prediction is the mean probability over their windows, so every
participant weighs once in the metrics, exactly as in ``classify.run``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGNet
from sklearn.model_selection import GroupShuffleSplit

from dementia_ml.eeg import io
from dementia_ml.eeg.classify import TASKS
from dementia_ml.eval import bootstrap, metrics
from dementia_ml.splits import subject_folds


def make_windows(
    bids_root: Path, sfreq: float = 125.0, window_s: float = 4.0, use_derivatives: bool = True
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Non-overlapping windows (n, channels, times), z-scored per channel per recording."""
    X, labels, subjects = [], [], []
    for row in io.participants(bids_root).itertuples():
        raw = io.load_subject(bids_root, row.subject, use_derivatives)
        raw.resample(sfreq, verbose=False)
        data = raw.get_data()
        data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        n = int(window_s * sfreq)
        for k in range(data.shape[1] // n):
            X.append(data[:, k * n : (k + 1) * n])
            labels.append(row.group)
            subjects.append(row.subject)
    return np.asarray(X, dtype=np.float32), pd.Series(labels), pd.Series(subjects)


def fold_indices(subjects: pd.Series, y: pd.Series, n_splits: int = 5, seed: int = 0):
    """(train, validation, test) row indices; no participant appears in two of them."""
    out = []
    for train, test in subject_folds(
        subjects.to_numpy(), n_splits, stratify=y.to_numpy(), seed=seed
    ):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, va = next(gss.split(train, groups=subjects.iloc[train].to_numpy()))
        out.append((train[tr], train[va], test))
    return out


def _fit_predict(Xtr, ytr, Xva, yva, Xte, n_classes, seed, max_epochs, patience, batch_size, sfreq):
    torch.manual_seed(seed)
    model = EEGNet(
        n_chans=Xtr.shape[1],
        n_outputs=n_classes,
        n_times=Xtr.shape[2],
        sfreq=sfreq,
        kernel_length=int(sfreq // 2),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait, history = np.inf, None, 0, []
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(total / len(perm))
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xva_t), yva_t))
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(Xte)), dim=1).numpy()
    return prob, history


def run(
    X: np.ndarray,
    labels: pd.Series,
    subjects: pd.Series,
    task: str,
    seed: int = 0,
    max_epochs: int = 15,
    patience: int = 4,
    n_splits: int = 5,
    batch_size: int = 64,
    sfreq: float = 125.0,
    n_boot: int = 1000,
) -> dict:
    classes, positive = TASKS[task]
    keep = labels.isin(classes).to_numpy()
    X, lab = X[keep], labels[keep].reset_index(drop=True)
    sub = subjects[keep].reset_index(drop=True)
    y = (
        (lab == positive).astype(int)
        if positive
        else lab.map({c: i for i, c in enumerate(classes)})
    )
    y_arr = y.to_numpy().astype(np.int64)
    n_classes = len(classes) if not positive else 2
    window_prob = np.zeros((len(X), n_classes))
    first, last = [], []
    for train, val, test in fold_indices(sub, y, n_splits, seed):
        prob, hist = _fit_predict(
            X[train],
            y_arr[train],
            X[val],
            y_arr[val],
            X[test],
            n_classes,
            seed,
            max_epochs,
            patience,
            batch_size,
            sfreq,
        )
        window_prob[test] = prob
        first.append(hist[0])
        last.append(hist[-1])
    per_subject = pd.DataFrame(window_prob, index=sub).groupby(level=0).mean()
    subj_y = y.groupby(sub).first().loc[per_subject.index].to_numpy()
    prob = per_subject.iloc[:, 1].to_numpy() if positive else per_subject.to_numpy()

    def auc(i):
        if positive:
            return metrics.auc(subj_y[i], prob[i])
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(subj_y[i], prob[i], multi_class="ovr", labels=range(len(classes)))
        )

    def bacc(i):
        pred = (prob[i] >= 0.5).astype(int) if positive else prob[i].argmax(axis=1)
        return metrics.balanced_accuracy(subj_y[i], pred)

    def with_ci(fn):
        est, lo, hi = bootstrap.ci(fn, n=len(subj_y), n_boot=n_boot, seed=seed)
        return {"estimate": est, "ci_low": lo, "ci_high": hi}

    preds = pd.DataFrame(
        {"prob": prob} if positive else {f"prob_{c}": prob[:, k] for k, c in enumerate(classes)},
        index=per_subject.index,
    ).assign(label=lab.groupby(sub).first().loc[per_subject.index].to_numpy())
    return {
        "task": task,
        "model": "eegnet",
        "classes": classes,
        "n_subjects": int(len(subj_y)),
        "n_windows": int(len(X)),
        "auc": with_ci(auc),
        "balanced_accuracy": with_ci(bacc),
        "train_loss_first": float(np.mean(first)),
        "train_loss_last": float(np.mean(last)),
        "subject_predictions": preds,
    }
