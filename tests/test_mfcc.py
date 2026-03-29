"""
Tests for the MFCC extractor.

Run with:  python tests/test_mfcc.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.mfcc import hz_to_mel, mel_to_hz, mel_filterbank, dct_matrix, mfcc, delta

SR = 16000


def make_sine(freq, duration, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


# ── mel ↔ hz round-trip ──────────────────────────────────────────────────────
def test_mel_hz_roundtrip():
    freqs = np.array([0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0])
    err   = np.max(np.abs(mel_to_hz(hz_to_mel(freqs)) - freqs))
    assert err < 1e-8, f"mel/hz round-trip error: {err:.2e}"
    print(f"[PASS] mel/hz round-trip  max_err={err:.2e}")


# ── filterbank shape & energy partition ──────────────────────────────────────
def test_filterbank():
    frame_len = 400
    fb = mel_filterbank(frame_len, SR, n_mels=26)

    assert fb.shape == (26, 201), f"fb shape mismatch: {fb.shape}"

    # Each filter should sum to something > 0
    row_sums = fb.sum(axis=1)
    assert np.all(row_sums > 0), "A filter has zero total weight"

    # All values non-negative
    assert np.all(fb >= 0), "Negative filter value found"

    print(f"[PASS] Filterbank  shape={fb.shape}, min_row_sum={row_sums.min():.4f}")


# ── DCT orthonormality ────────────────────────────────────────────────────────
def test_dct_orthonormal():
    D = dct_matrix(26, 13)
    # D @ D.T should be identity (rows are orthonormal)
    gram = D @ D.T
    err  = np.max(np.abs(gram - np.eye(13)))
    assert err < 1e-12, f"DCT rows not orthonormal: max dev={err:.2e}"
    print(f"[PASS] DCT orthonormality  max_dev={err:.2e}")


# ── MFCC output shape ─────────────────────────────────────────────────────────
def test_mfcc_shape():
    signal = make_sine(440, 1.0)
    coeffs = mfcc(signal, SR, n_mfcc=13, n_mels=26)

    # Should have 13 coefficients per frame
    assert coeffs.shape[1] == 13, f"Expected 13 coefficients, got {coeffs.shape[1]}"
    assert coeffs.shape[0] > 0,   "Zero frames returned"
    print(f"[PASS] MFCC shape  {coeffs.shape}  ({coeffs.shape[0]} frames, 13 coeffs)")


# ── MFCCs differ between signals ──────────────────────────────────────────────
def test_mfcc_discriminates():
    sig_440  = make_sine(440,  1.0)
    sig_2000 = make_sine(2000, 1.0)

    c1 = mfcc(sig_440,  SR).mean(axis=0)
    c2 = mfcc(sig_2000, SR).mean(axis=0)

    dist = np.linalg.norm(c1 - c2)
    assert dist > 1.0, f"MFCCs too similar for different signals: dist={dist:.4f}"
    print(f"[PASS] MFCC discrimination  ||c_440 - c_2000||={dist:.4f}")


# ── delta shape & symmetry ────────────────────────────────────────────────────
def test_delta():
    signal = make_sine(440, 1.0)
    coeffs = mfcc(signal, SR)
    d1     = delta(coeffs)
    d2     = delta(d1)        # delta-delta

    assert d1.shape == coeffs.shape, "delta shape mismatch"
    assert d2.shape == coeffs.shape, "delta-delta shape mismatch"

    # For a stationary signal (sine), deltas should be near zero on average
    mean_abs = np.abs(d1).mean()
    assert mean_abs < 1.0, f"Delta mean too large for stationary signal: {mean_abs:.4f}"
    print(f"[PASS] Delta  shape={d1.shape}  mean_abs={mean_abs:.6f}")


if __name__ == "__main__":
    test_mel_hz_roundtrip()
    test_filterbank()
    test_dct_orthonormal()
    test_mfcc_shape()
    test_mfcc_discriminates()
    test_delta()
    print("\nAll tests passed.")
