"""
src/analysis/wm_spectrum.py

Watermark-perturbation analysis for the DeepMark benchmark.

Reuses the existing plugin layer: models are reached over HTTP through their
BaseModel subclass, attacks are reached through PluginManager. Nothing here
imports torch or audioseal -- that all stays inside the Docker service, so this
runs in the plain benchmark venv from a notebook at the repo root.

    docker compose up -d audioseal

    import sys; sys.path.insert(0, "src")
    from analysis import wm_spectrum as ws
"""

import logging
import os
import sys

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

MODEL_SR = 16000
EPS = 1e-12


# --------------------------------------------------------------------------- #
# plugin access
# --------------------------------------------------------------------------- #

def _ensure_src_on_path():
    """Make `src/` importable whether the notebook sits at repo root or in src/."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.dirname(here)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_plugins():
    """Return (models, attacks) dicts from PluginManager.

    Note: PluginManager imports EVERY plugin at startup. A plugin with a
    missing dependency fails silently and just won't appear in the dict.
    """
    _ensure_src_on_path()
    from plugin_manager import PluginManager

    pm = PluginManager()
    return pm.get_models(), pm.get_attacks()


def get_model(name="AudioSealModel"):
    models, _ = load_plugins()
    if name not in models:
        raise KeyError(f"{name} not discovered. Available: {sorted(models)}")
    return models[name]["class"]()


def get_attack(name):
    _, attacks = load_plugins()
    if name not in attacks:
        raise KeyError(f"{name} not discovered. Available: {sorted(attacks)}")
    entry = attacks[name]
    return entry["class"](), (entry["config"] or {})


def apply_attack(name, audio, sampling_rate, **overrides):
    """Run one benchmark attack with its config.json defaults, overridable."""
    attack, defaults = get_attack(name)
    params = {**defaults, **overrides}
    return np.asarray(attack.apply(audio, sampling_rate=sampling_rate, **params))


# --------------------------------------------------------------------------- #
# residual extraction
# --------------------------------------------------------------------------- #

def watermark_residual(model, audio, sampling_rate, bits, mode="native"):
    """The additive perturbation itself, via the /watermark endpoint.

    Requires the endpoint added to the audio_seal plugin's app.py.
    """
    payload = {
        "audio": np.asarray(audio, dtype=float).tolist(),
        "watermark_data": np.asarray(bits, dtype=int).tolist(),
        "sampling_rate": int(sampling_rate),
        "mode": mode,
    }
    resp = model._make_request(endpoint="/watermark", json_data=payload)
    return np.asarray(resp["watermark_signal"])


def watermark_residual_by_subtraction(model, audio, sampling_rate, bits):
    """Fallback using only the stock /embed endpoint.

    ONLY valid at 16 kHz. Above that, /embed's resample round-trip strips
    content over 8 kHz, and the difference is dominated by that loss rather
    than by the watermark.
    """
    if sampling_rate != MODEL_SR:
        logger.warning(
            "subtraction at %d Hz mixes the watermark with resampling loss; "
            "use watermark_residual() instead", sampling_rate
        )
    y = model.embed(np.asarray(audio), np.asarray(bits), sampling_rate)
    return np.asarray(y)[: len(audio)] - np.asarray(audio)[: len(y)]


# --------------------------------------------------------------------------- #
# signal analysis
# --------------------------------------------------------------------------- #

def stft_db(x, sr, nperseg=1024, overlap=0.75):
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=nperseg,
                   noverlap=int(nperseg * overlap), window="hann")
    return f, t, 20.0 * np.log10(np.abs(Z) + EPS)


def mean_spectrum_db(x, sr, nperseg=1024, overlap=0.75):
    """Long-term average spectrum. Averaged in linear magnitude, not in dB."""
    from scipy.signal import stft
    f, _, Z = stft(x, fs=sr, nperseg=nperseg,
                   noverlap=int(nperseg * overlap), window="hann")
    return f, 20.0 * np.log10(np.abs(Z).mean(axis=1) + EPS)


def si_snr(host, residual):
    """SI-SNR of host vs host+residual -- the metric the AudioSeal paper reports."""
    s = np.asarray(host, dtype=float)
    sw = s + np.asarray(residual, dtype=float)
    alpha = np.dot(s, sw) / (np.dot(s, s) + EPS)
    return 10.0 * np.log10((np.sum((alpha * s) ** 2) + EPS) /
                           (np.sum((alpha * s - sw) ** 2) + EPS))


def hf_energy_fraction(residual, sr, cutoff=8000.0):
    """Share of watermark energy above `cutoff`. High == codec-fragile."""
    from scipy.signal import stft
    f, _, Z = stft(residual, fs=sr, nperseg=1024, noverlap=768, window="hann")
    p = (np.abs(Z) ** 2).sum(axis=1)
    return float(p[f > cutoff].sum() / (p.sum() + EPS))


def bit_accuracy(sent, recovered):
    """Percentage, 0-100, matching the benchmark's convention."""
    sent, recovered = np.asarray(sent), np.asarray(recovered)
    if recovered.size != sent.size:
        return 0.0
    return 100.0 * float((sent == recovered).mean())


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def plot_watermark(host, residuals, sr, figsize_per_col=5.5):
    """host: 1-D array. residuals: {label: 1-D array}. Returns the figure."""
    import matplotlib.pyplot as plt

    n = len(residuals)
    fig, axes = plt.subplots(2, 1 + n, figsize=(figsize_per_col * (1 + n), 8),
                             constrained_layout=True, squeeze=False)

    f, t, S = stft_db(host, sr)
    axes[0][0].pcolormesh(t, f / 1000, S, shading="gouraud", cmap="magma",
                          vmin=S.max() - 80, vmax=S.max())
    axes[0][0].set_title("Host")
    axes[0][0].set_ylabel("kHz")

    for i, (label, r) in enumerate(residuals.items(), start=1):
        f, t, W = stft_db(r, sr)
        # Normalised to the RESIDUAL's own peak. It sits ~26 dB under the host,
        # so on an absolute scale it renders as an empty black rectangle.
        im = axes[0][i].pcolormesh(t, f / 1000, W, shading="gouraud", cmap="magma",
                                   vmin=W.max() - 60, vmax=W.max())
        axes[0][i].set_title(f"Residual — {label}")
        fig.colorbar(im, ax=axes[0][i], label="dB rel. own peak")
    for ax in axes[0]:
        ax.set_xlabel("s")

    fh, Sh = mean_spectrum_db(host, sr)
    ax = axes[1][0]
    ax.plot(fh / 1000, Sh, color="0.45", lw=1, label="host")
    for label, r in residuals.items():
        fw, Sw = mean_spectrum_db(r, sr)
        ax.plot(fw / 1000, Sw, lw=1.4, label=label)
    if sr > 2 * MODEL_SR:
        ax.axvline(MODEL_SR / 2000, color="crimson", ls="--", lw=1, label="8 kHz")
    ax.set(xlabel="kHz", ylabel="dB", title="Long-term average spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    for i, (label, r) in enumerate(residuals.items(), start=1):
        fw, Sw = mean_spectrum_db(r, sr)
        axes[1][i].plot(fw / 1000, Sw - Sh, lw=1.4, color="steelblue")
        axes[1][i].axhline(0, color="k", lw=0.6)
        if sr > 2 * MODEL_SR:
            axes[1][i].axvline(MODEL_SR / 2000, color="crimson", ls="--", lw=1)
        axes[1][i].set(xlabel="kHz", ylabel="residual − host (dB)",
                       title=f"Watermark-to-signal ratio — {label}")
        axes[1][i].grid(alpha=0.25)

    return fig


# --------------------------------------------------------------------------- #
# end-to-end sweep
# --------------------------------------------------------------------------- #

def load_wav(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x[:, 0], sr


def sweep(wav_path, attack_names, model_name="AudioSealModel",
          modes=("native", "banded"), seed=0, **attack_overrides):
    """Embed under each mode, run each attack, detect. Returns list of dicts.

    Feed to pandas.DataFrame(...) in the notebook.
    """
    rng = np.random.default_rng(seed)
    model = get_model(model_name)
    host, sr = load_wav(wav_path)
    bits = rng.integers(0, 2, size=model.config["watermark_size"], dtype=np.int32)

    rows = []
    for mode in modes:
        wm = watermark_residual(model, host, sr, bits, mode=mode)
        watermarked = host + wm
        base = {
            "file": os.path.basename(wav_path),
            "mode": mode,
            "si_snr_db": si_snr(host, wm),
            "hf_frac_8k": hf_energy_fraction(wm, sr),
        }
        for attack_name in ["none", *attack_names]:
            y = (watermarked if attack_name == "none"
                 else apply_attack(attack_name, watermarked, sr, **attack_overrides))
            got, conf = model.detect(np.asarray(y), sr)
            rows.append({**base, "attack": attack_name,
                         "bit_acc": bit_accuracy(bits, got),
                         "confidence": float(conf)})
    return rows