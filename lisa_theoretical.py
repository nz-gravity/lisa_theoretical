#!/usr/bin/env python3
"""
Produce analytic LISA PSDs/CSDs using the LDC noise realization and compare them
against periodograms estimated from the provided tdi.h5 file.  The script always:

1. Loads X2/Y2/Z2 time series from data/tdi.h5
2. Estimates the auto- and cross-periodograms
3. Builds the TDI2 analytic PSD/CSD using the LDC discrete-noise filters
4. Saves the PSD/CSD table + covariance cube
5. Generates a triangle plot with PSDs on the diagonal and |CSD|s off-diagonal
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


# --- Constants and default paths -------------------------------------------------
C_LIGHT = 299_792_458.0  # m / s
L_ARM = 2.5e9  # m
LIGHT_TRAVEL_TIME = L_ARM / C_LIGHT  # ≈ 8.33 s

OMS_ASD = 1.5e-11
OMS_FKNEE = 2e-3
PM_ASD = 3e-15
PM_LOW_FKNEE = 4e-4
PM_HIGH_FKNEE = 8e-3
LASER_FREQ = 2.81e14  # Hz

DATA_PATH = Path("data/tdi.h5")
PSD_PNG = Path("lisa_psd_plot.png")
TRIANGLE_PNG = Path("spectra_triangle.png")


# --- Noise and transfer helpers --------------------------------------------------
def lisa_link_noises_ldc(
    freq: np.ndarray,
    fs: float,
    fmin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the single-link proof-mass (Spm) and optical-path (Sop) PSDs
    used in the LDC noise realizations.

    See https://zenodo.org/doi/10.5281/zenodo.15698080
    """
    exp_term = np.exp(-2.0 * np.pi * fmin / fs) * np.exp(-2j * np.pi * freq / fs)
    denom_mag2 = np.abs(1.0 - exp_term) ** 2

    psd_tm_high = (
        (2.0 * PM_ASD * LASER_FREQ / (2.0 * np.pi * C_LIGHT)) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    psd_tm_low = (
        (2.0 * PM_ASD * LASER_FREQ * PM_LOW_FKNEE / (2.0 * np.pi * C_LIGHT)) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
        * np.abs(1.0 / (1.0 - np.exp(-2j * np.pi * freq / fs))) ** 2
        * (2.0 * np.pi / fs) ** 2
    )
    Spm = psd_tm_high + psd_tm_low

    psd_oms_high = (OMS_ASD * fs * LASER_FREQ / C_LIGHT) ** 2 * np.sin(
        2.0 * np.pi * freq / fs
    ) ** 2
    psd_oms_low = (
        (2.0 * np.pi * OMS_ASD * LASER_FREQ * OMS_FKNEE**2 / C_LIGHT) ** 2
        * (2.0 * np.pi * fmin) ** 2
        / denom_mag2
        / (fs * fmin) ** 2
    )
    Sop = psd_oms_high + psd_oms_low
    return Spm, Sop


def tdi2_psd_and_csd(freq: np.ndarray, Spm: np.ndarray, Sop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diagonal PSD (X2) and cross-term CSD (XY) for TDI2 combinations.

    Because of the symmetries of the equal-arm constellation:
        S_X2 = S_Y2 = S_Z2
        S_XY = S_YZ = S_ZX
    """
    x = 2.0 * np.pi * LIGHT_TRAVEL_TIME * freq
    sinx = np.sin(x)
    sin2x = np.sin(2.0 * x)
    cosx = np.cos(x)
    cos2x = np.cos(2.0 * x)

    diag = 64.0 * sinx**2 * sin2x**2 * Sop
    diag += 256.0 * (3.0 + cos2x) * cosx**2 * sinx**4 * Spm

    csd = -16.0 * sinx * (sin2x**3) * (4.0 * Spm + Sop)
    return diag, csd


def covariance_matrix(diag: np.ndarray, csd: np.ndarray) -> np.ndarray:
    """Assemble the 3×3 covariance matrix Σ(f) for each frequency."""
    nf = diag.size
    cov = np.zeros((nf, 3, 3))
    cov[:, 0, 0] = cov[:, 1, 1] = cov[:, 2, 2] = diag
    cov[:, 0, 1] = cov[:, 1, 0] = csd
    cov[:, 1, 2] = cov[:, 2, 1] = csd
    cov[:, 0, 2] = cov[:, 2, 0] = csd
    return cov


def make_plot(freq: np.ndarray, diag: np.ndarray, csd: np.ndarray, out_path: Path) -> None:
    """Produce a quick-look log-log plot of PSD and |CSD|."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(freq, diag, label="PSD (X2=Y2=Z2)")
    ax.loglog(freq, np.abs(csd), label="|CSD| (XY=YZ=ZX)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectral Density [1/Hz]")
    ax.legend()
    ax.set_title("LISA TDI2 analytic noise")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --- Data handling + spectral estimates -----------------------------------------
def load_tdi_timeseries(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read t, X2, Y2, Z2 arrays from the HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        t = np.array(f["t"])
        X2 = np.array(f["X2"])
        Y2 = np.array(f["Y2"])
        Z2 = np.array(f["Z2"])
    return t, X2, Y2, Z2


def compute_periodograms(
    t: np.ndarray,
    X2: np.ndarray,
    Y2: np.ndarray,
    Z2: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Return (freq, auto-PSD dict, cross-CSD dict, metadata) from the time-domain data."""
    dt = t[1] - t[0]
    n = len(t)
    freq_full = np.fft.rfftfreq(n, dt)
    fft_x = np.fft.rfft(X2)
    fft_y = np.fft.rfft(Y2)
    fft_z = np.fft.rfft(Z2)

    scale = dt / n
    auto = {
        "X": scale * np.abs(fft_x) ** 2,
        "Y": scale * np.abs(fft_y) ** 2,
        "Z": scale * np.abs(fft_z) ** 2,
    }
    cross = {
        "XY": scale * fft_x * np.conj(fft_y),
        "YZ": scale * fft_y * np.conj(fft_z),
        "ZX": scale * fft_z * np.conj(fft_x),
    }

    # Double the positive-frequency interior to account for two-sided FFT.
    for arr in list(auto.values()) + list(cross.values()):
        arr[1:-1] *= 2.0

    # Drop the DC bin for plotting (log axis incompatible with zero frequency).
    freq = freq_full[1:]
    for key in auto:
        auto[key] = auto[key][1:]
    for key in cross:
        cross[key] = cross[key][1:]

    meta = {"dt": dt, "fs": 1.0 / dt, "n": n, "fmin": 1.0 / (n * dt)}
    return freq, auto, cross, meta


# --- Plotting --------------------------------------------------------------------
def plot_triangle_spectra(
    freq: np.ndarray,
    diag: np.ndarray,
    csd: np.ndarray,
    auto_psd: Dict[str, np.ndarray],
    cross_csd: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """
    Plot a lower-triangular matrix with PSDs on the diagonal and |CSD|s
    on the off-diagonal entries:

        X
        |XY|   Y
        |XZ|  |YZ|   Z
    """

    channels = ["X", "Y", "Z"]
    combo_map = {
        (1, 0): "XY",
        (2, 0): "ZX",  # equals |XZ|
        (2, 1): "YZ",
    }

    fig, axes = plt.subplots(3, 3, figsize=(8.0, 8.0), sharex=True, sharey=False)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            if i == j:
                chan = channels[i]
                ax.loglog(freq, auto_psd.get(chan, diag), label=f"{chan} periodogram", color=f"C{i}", alpha=0.7)
                ax.loglog(freq, diag, "k-", lw=1.2, label="Analytic PSD")
                ax.set_title(f"{chan} PSD")
            else:
                combo = combo_map[(i, j)]
                label = f"|{combo}| periodogram"
                ax.loglog(freq, np.abs(cross_csd.get(combo, csd)), label=label, color="tab:blue", alpha=0.7)
                ax.loglog(freq, np.abs(csd), "k-", lw=1.2, label="|Analytic CSD|")
                ax.set_title(f"|{combo}| CSD")

            ax.set_yscale("log")
            ax.set_xscale("log")
            if i == 2:
                ax.set_xlabel("Frequency [Hz]")
            if j == 0:
                ax.set_ylabel("Spectral density [1/Hz]")
            ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --- Main entry point ------------------------------------------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Download the LDC tdi.h5 file first.")

    t, X2, Y2, Z2 = load_tdi_timeseries(DATA_PATH)
    freq, auto_psd, cross_csd, meta = compute_periodograms(t, X2, Y2, Z2)
    print(f"Loaded {len(t)} samples from {DATA_PATH} (fs={meta['fs']:.6f} Hz).")

    Spm, Sop = lisa_link_noises_ldc(freq, fs=meta["fs"], fmin=meta["fmin"])
    diag, csd = tdi2_psd_and_csd(freq, Spm, Sop)
    cov = covariance_matrix(diag, csd)

    make_plot(freq, diag, csd, PSD_PNG)
    plot_triangle_spectra(freq, diag, csd, auto_psd, cross_csd, TRIANGLE_PNG)

    print(f"Wrote analytic PSD plot to {PSD_PNG.resolve()}")
    print(f"Wrote triangle plot to {TRIANGLE_PNG.resolve()}")


if __name__ == "__main__":
    main()
