"""Spectral features of resting-state EEG, one row per epoch."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
FMIN, FMAX = 0.5, 45.0
ALPHA_SEARCH = (7.0, 13.0)


def epoch_features(raw: mne.io.BaseRaw, epoch_s: float = 4.0, overlap: float = 0.5) -> pd.DataFrame:
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_s, overlap=epoch_s * overlap, preload=True, verbose=False
    )
    sfreq = raw.info["sfreq"]
    n_fft = int(2 * sfreq)  # 2-s Welch segments -> 0.5 Hz resolution
    spec = epochs.compute_psd(
        method="welch", fmin=FMIN, fmax=FMAX, n_fft=n_fft, n_overlap=n_fft // 2, verbose=False
    )
    psd, freqs = spec.get_data(return_freqs=True)  # (epochs, channels, freqs)
    total = psd.sum(axis=-1)
    cols: dict[str, np.ndarray] = {}
    rel = {}
    for band, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & ((freqs < hi) | ((hi == FMAX) & (freqs <= hi)))
        rel[band] = psd[..., m].sum(axis=-1) / total
    p = psd / total[..., None]
    entropy = -(p * np.log(p)).sum(axis=-1) / np.log(p.shape[-1])
    for c, ch in enumerate(epochs.ch_names):
        for band in BANDS:
            cols[f"{ch}_rel_{band}"] = rel[band][:, c]
        cols[f"{ch}_theta_alpha"] = rel["theta"][:, c] / rel["alpha"][:, c]
        cols[f"{ch}_spectral_entropy"] = entropy[:, c]
    am = (freqs >= ALPHA_SEARCH[0]) & (freqs <= ALPHA_SEARCH[1])
    mean_psd = psd.mean(axis=1)[:, am]
    cols["peak_alpha_hz"] = freqs[am][mean_psd.argmax(axis=-1)]
    return pd.DataFrame(cols)
