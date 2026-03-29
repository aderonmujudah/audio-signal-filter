"""
Tests for the LPC + formant extractor.

Run with:  python tests/test_lpc.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.lpc import (
    pre_emphasis, autocorrelation, levinson_durbin, lpc, lpc_to_formants, formants
)

SR = 16000


def make_sine(freq, duration=1.0, sr=SR, amp=1.0):
    t = np.arange(int(duration * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def make_voiced(f0=120, formant_freqs=(800, 1200, 2500), duration=0.5, sr=SR):
    """Synthesise a vowel-like signal: harmonic stack shaped by simple resonances."""
    t = np.arange(int(duration * sr)) / sr
    # Harmonic source
    signal = np.zeros_like(t)
    for k in range(1, 20):
        signal += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    # Rough resonance shaping via summed sines (not a real formant filter, but
    # enough to put spectral energy near the target frequencies for testing)
    for ff in formant_freqs:
        signal += 0.5 * np.sin(2 * np.pi * ff * t)
    return signal / np.max(np.abs(signal))


# ── pre-emphasis ─────────────────────────────────────────────────────────────
def test_pre_emphasis():
    signal = np.ones(100)
    out = pre_emphasis(signal, 0.97)
    # First sample unchanged, rest should be 1 - 0.97*1 = 0.03
    assert abs(out[0] - 1.0) < 1e-12
    assert np.allclose(out[1:], 0.03)
    print("[PASS] pre_emphasis")


# ── autocorrelation R[0] == power ────────────────────────────────────────────
def test_autocorrelation():
    frame = np.random.default_rng(0).standard_normal(400)
    R = autocorrelation(frame, order=16)

    # R[0] should equal mean power
    expected_R0 = np.dot(frame, frame) / len(frame)
    assert abs(R[0] - expected_R0) < 1e-12, f"R[0] mismatch: {R[0]:.6f} vs {expected_R0:.6f}"

    # Toeplitz symmetry: R[k] == R[-k] (biased estimator is symmetric)
    R_full = autocorrelation(frame, order=4)
    assert len(R_full) == 5
    print(f"[PASS] autocorrelation  R[0]={R[0]:.4f}")


# ── Levinson-Durbin: round-trip via prediction error ─────────────────────────
def test_levinson_durbin():
    # Build an AR(2) process and check that LD recovers approximately the right
    # coefficients from a long frame.
    rng = np.random.default_rng(42)
    N   = 4000
    # AR(2): s[n] = 1.3*s[n-1] - 0.8*s[n-2] + noise
    true_a = np.array([1.3, -0.8])   # note sign: s[n] + a[1]*s[n-1] + a[2]*s[n-2] = e
    # Equivalent: prediction coefficients are [1.3, -0.8] in our notation
    x = np.zeros(N)
    e = rng.standard_normal(N) * 0.1
    for n in range(2, N):
        x[n] = true_a[0] * x[n-1] + true_a[1] * x[n-2] + e[n]

    R = autocorrelation(x, order=2)
    a = levinson_durbin(R)   # should recover ~ [-1.3, 0.8] (our sign convention)

    # In our convention: x[n] = -a[0]*x[n-1] - a[1]*x[n-2] + e[n]
    # => a[0] = -1.3, a[1] = 0.8
    assert abs(a[0] - (-true_a[0])) < 0.05, f"a[0] off: {a[0]:.4f} vs {-true_a[0]:.4f}"
    assert abs(a[1] - (-true_a[1])) < 0.05, f"a[1] off: {a[1]:.4f} vs {-true_a[1]:.4f}"
    print(f"[PASS] Levinson-Durbin  a={a[:2]}  (expected ~{[-true_a[0], -true_a[1]]})")


# ── LPC output shape ─────────────────────────────────────────────────────────
def test_lpc_shape():
    signal = make_sine(440)
    order  = 14
    A, frame_len, hop_len = lpc(signal, SR, order=order)

    assert A.shape[1] == order, f"Expected {order} coefficients per frame"
    assert A.shape[0] > 0
    print(f"[PASS] LPC shape  {A.shape}")


# ── Formant frequencies plausible for a vowel-like signal ────────────────────
def test_formants_plausible():
    signal = make_voiced(f0=120, formant_freqs=(800, 1200, 2500))
    f, bw  = formants(signal, SR, order=14, n_formants=3)

    # At least half the frames should have a valid F1
    valid_f1 = np.sum(~np.isnan(f[:, 0]))
    assert valid_f1 > f.shape[0] * 0.4, f"Too few valid F1 frames: {valid_f1}/{f.shape[0]}"

    # Mean F1 should be in a rough speech range (200–1500 Hz)
    mean_f1 = np.nanmean(f[:, 0])
    assert 200 < mean_f1 < 1500, f"Mean F1 implausible: {mean_f1:.1f} Hz"

    print(f"[PASS] Formants  mean F1={mean_f1:.0f} Hz  "
          f"mean F2={np.nanmean(f[:,1]):.0f} Hz  "
          f"valid_f1={valid_f1}/{f.shape[0]}")


# ── Silent frame → all NaN formants (no crash) ───────────────────────────────
def test_silent_frame():
    silence = np.zeros(SR)
    f, bw   = formants(silence, SR)
    assert np.all(np.isnan(f)), "Expected all-NaN formants for silence"
    print("[PASS] silent frame -> all-NaN formants")


if __name__ == "__main__":
    test_pre_emphasis()
    test_autocorrelation()
    test_levinson_durbin()
    test_lpc_shape()
    test_formants_plausible()
    test_silent_frame()
    print("\nAll tests passed.")
