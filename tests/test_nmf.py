"""
Tests for NMF and Griffin-Lim.

Run with:  python tests/test_nmf.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.nmf   import nmf, griffin_lim
from audio_filter.stft  import stft, magnitude_phase

SR        = 16000
FRAME_LEN = 400   # 25 ms at 16 kHz
HOP_LEN   = 160   # 10 ms at 16 kHz
N_BINS    = FRAME_LEN // 2 + 1   # 201


def make_magnitude(n_bins=N_BINS, n_frames=120, n_components=2, seed=0):
    """Synthetic magnitude spectrogram as W_true @ H_true (rank-n_components)."""
    rng = np.random.default_rng(seed)
    W_true = rng.random((n_bins, n_components)) + 0.1
    H_true = rng.random((n_components, n_frames)) + 0.1
    return W_true @ H_true


# ── output shapes ─────────────────────────────────────────────────────────────
def test_nmf_shape():
    V = make_magnitude()
    n_bins, n_frames = V.shape
    W, H = nmf(V, n_components=3)
    assert W.shape == (n_bins, 3),    f"W shape: {W.shape}"
    assert H.shape == (3, n_frames),  f"H shape: {H.shape}"
    print(f"[PASS] nmf shape  W={W.shape}  H={H.shape}")


# ── W and H are non-negative ──────────────────────────────────────────────────
def test_nmf_nonnegative():
    V = make_magnitude()
    W, H = nmf(V, n_components=2)
    assert np.all(W >= 0), f"W has negative values: min={W.min():.4e}"
    assert np.all(H >= 0), f"H has negative values: min={H.min():.4e}"
    print(f"[PASS] NMF non-negative  W_min={W.min():.2e}  H_min={H.min():.2e}")


# ── reconstruction error < 5 % of input norm ─────────────────────────────────
def test_nmf_reconstruction_quality():
    V  = make_magnitude(n_components=2)
    W, H = nmf(V, n_components=2, max_iter=500, tol=1e-6)
    err  = np.linalg.norm(V - W @ H, "fro") / np.linalg.norm(V, "fro")
    assert err < 0.05, f"Reconstruction error too large: {err:.4f}"
    print(f"[PASS] NMF reconstruction  rel_err={err:.4f}")


# ── rank-1 approximation captures the dominant structure ─────────────────────
def test_nmf_rank1():
    V = make_magnitude(n_components=1)   # rank-1 ground truth
    W, H = nmf(V, n_components=1, max_iter=500, tol=1e-6)
    err  = np.linalg.norm(V - W @ H, "fro") / np.linalg.norm(V, "fro")
    assert err < 0.02, f"Rank-1 NMF error too large: {err:.4f}"
    print(f"[PASS] NMF rank-1  rel_err={err:.4f}")


# ── spectral separation: two band-limited sources ─────────────────────────────
def test_nmf_spectral_separation():
    """
    V = W_true @ H_true where W_true has energy concentrated in two
    disjoint frequency bands.  After NMF, each W column should be
    concentrated in one of those bands.
    """
    rng = np.random.default_rng(7)
    n_bins   = 64
    n_frames = 200
    n_low    = 16   # source 0 occupies bins 0 .. 15
    n_high   = 16   # source 1 occupies bins 48 .. 63

    W_true = np.zeros((n_bins, 2))
    W_true[:n_low, 0]   = rng.random(n_low)   + 0.5
    W_true[-n_high:, 1] = rng.random(n_high)  + 0.5

    H_true = rng.random((2, n_frames)) + 0.5
    V = W_true @ H_true + 1e-3 * rng.random((n_bins, n_frames))

    W, H = nmf(V, n_components=2, max_iter=500, tol=1e-6)

    # Normalize W columns to sum = 1 for comparison
    W_norm = W / (W.sum(axis=0, keepdims=True) + 1e-10)

    # For each recovered component, measure fraction of energy in each band
    low_frac  = [W_norm[:n_low, k].sum()   for k in range(2)]
    high_frac = [W_norm[-n_high:, k].sum() for k in range(2)]

    # One component should be mostly low, the other mostly high
    assert max(low_frac)  > 0.5, f"No component concentrated in low band:  {low_frac}"
    assert max(high_frac) > 0.5, f"No component concentrated in high band: {high_frac}"
    print(f"[PASS] NMF spectral separation  "
          f"low_frac={[f'{v:.2f}' for v in low_frac]}  "
          f"high_frac={[f'{v:.2f}' for v in high_frac]}")


# ── Griffin-Lim: output is 1-D ────────────────────────────────────────────────
def test_griffin_lim_shape():
    V = make_magnitude()
    sig = griffin_lim(V, FRAME_LEN, HOP_LEN, SR, n_iter=4)
    assert sig.ndim == 1,   f"Expected 1-D signal, got shape {sig.shape}"
    assert len(sig) > 0,    "Griffin-Lim returned empty signal"
    print(f"[PASS] Griffin-Lim shape  len={len(sig)}")


# ── Griffin-Lim: |STFT(output)| converges toward input magnitude ─────────────
def test_griffin_lim_magnitude_consistency():
    """
    After sufficient iterations, the magnitude of STFT(GL output) should
    be close to the target magnitude (normalized mean absolute error < 0.10).
    """
    # Use a real signal so the target magnitude is actually achievable
    t      = np.arange(SR) / SR
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    S, fl, hl = stft(signal, SR,
                     frame_sec=FRAME_LEN / SR,
                     hop_sec=HOP_LEN   / SR)

    magnitude = np.abs(S).T          # (n_bins, n_frames)

    reconstructed = griffin_lim(magnitude, fl, hl, SR, n_iter=100, random_state=0)

    S_rec, _, _ = stft(reconstructed, SR,
                       frame_sec=fl / SR,
                       hop_sec=hl   / SR)

    mag_rec = np.abs(S_rec).T        # (n_bins, n_frames_new)
    min_f   = min(magnitude.shape[1], mag_rec.shape[1])
    mag_tgt = magnitude[:, :min_f]
    mag_got = mag_rec[:, :min_f]

    rel_err = np.linalg.norm(mag_got - mag_tgt, "fro") / (np.linalg.norm(mag_tgt, "fro") + 1e-10)
    assert rel_err < 0.10, f"Griffin-Lim magnitude error too large: rel_err={rel_err:.4f}"
    print(f"[PASS] Griffin-Lim magnitude consistency  rel_err={rel_err:.4f}")


# ── Griffin-Lim: improves with more iterations ────────────────────────────────
def test_griffin_lim_iteration_improvement():
    """More iterations should reduce magnitude error."""
    t      = np.arange(SR * 2) / SR
    signal = np.sin(2 * np.pi * 330 * t)
    S, fl, hl = stft(signal, SR,
                     frame_sec=FRAME_LEN / SR,
                     hop_sec=HOP_LEN   / SR)
    magnitude = np.abs(S).T

    def mag_error(n_iter):
        rec     = griffin_lim(magnitude, fl, hl, SR, n_iter=n_iter, random_state=0)
        S_r, _, _ = stft(rec, SR, frame_sec=fl / SR, hop_sec=hl / SR)
        m_r     = np.abs(S_r).T
        min_f   = min(magnitude.shape[1], m_r.shape[1])
        return np.linalg.norm(m_r[:, :min_f] - magnitude[:, :min_f], "fro")

    err_few  = mag_error(2)
    err_many = mag_error(32)
    assert err_many < err_few, (
        f"More iterations should reduce error: err(2)={err_few:.4e}  err(32)={err_many:.4e}"
    )
    print(f"[PASS] GL iteration improvement  err(2)={err_few:.4e}  err(32)={err_many:.4e}")


if __name__ == "__main__":
    test_nmf_shape()
    test_nmf_nonnegative()
    test_nmf_reconstruction_quality()
    test_nmf_rank1()
    test_nmf_spectral_separation()
    test_griffin_lim_shape()
    test_griffin_lim_magnitude_consistency()
    test_griffin_lim_iteration_improvement()
    print("\nAll tests passed.")
