import mne
import numpy as np
import pytest

from dementia_ml.eeg import features

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]


def _pink(n, rng):
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.arange(len(spec))
    spec[1:] /= np.sqrt(f[1:])
    return np.fft.irfft(spec, n)


def _raw(alpha_hz=10.0, alpha_amp=3.0, seconds=60, sfreq=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * sfreq) / sfreq
    data = (
        np.array(
            [
                alpha_amp * np.sin(2 * np.pi * alpha_hz * t + rng.uniform(0, 6))
                + _pink(len(t), rng) * 20
                for _ in CHANNELS
            ]
        )
        * 1e-6
    )
    info = mne.create_info(CHANNELS, sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture(scope="module")
def feats():
    return features.epoch_features(_raw(), epoch_s=4.0, overlap=0.5)


def test_alpha_peak_detected(feats):
    assert abs(feats["peak_alpha_hz"].median() - 10.0) <= 0.5
    rel = {b: feats[f"O1_rel_{b}"].mean() for b in features.BANDS}
    assert max(rel, key=rel.get) == "alpha"


def test_relative_powers_sum_to_one(feats):
    total = sum(feats[f"C3_rel_{b}"] for b in features.BANDS)
    assert np.allclose(total, 1.0, atol=1e-6)


def test_epoch_count_and_columns(feats):
    # 60 s, 4 s windows, 50 % overlap -> 29 epochs
    assert len(feats) == 29
    assert {"O1_theta_alpha", "O1_spectral_entropy", "peak_alpha_hz"} <= set(feats.columns)
