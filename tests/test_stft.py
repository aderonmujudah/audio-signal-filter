"""
Quick verification of the STFT / ISTFT engine.

Tests:
  1. Perfect reconstruction — STFT → ISTFT recovers the original signal.
  2. Shape check          — output dimensions match expected values.
  3. Sine-wave content    — a 440 Hz tone shows energy at the right bin.

Run with:  python -m pytest tests/test_stft.py -v
  or just: python tests/test_stft.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audio_filter.stft import stft, istft, magnitude_phase, reconstruct_from_magnitude

SR = 16000  # 16 kHz test sample rate


def make_sine(freq: float, duration: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


# ── Test 1: perfect reconstruction ──────────────────────────────────────────
def test_perfect_reconstruction():
    signal = make_sine(440, 1.0, SR)
    S, frame_len, hop_len = stft(signal, SR)
    reconstructed = istft(S, SR, frame_len, hop_len, original_len=len(signal))

    max_err = np.max(np.abs(signal - reconstructed))
    assert max_err < 1e-10, f"Reconstruction error too large: {max_err:.2e}"
    print(f"[PASS] Perfect reconstruction  max_err={max_err:.2e}")


# ── Test 2: output shape ─────────────────────────────────────────────────────
def test_shape():
    signal    = make_sine(440, 0.5, SR)
    frame_sec = 0.025
    hop_sec   = 0.010

    S, frame_len, hop_len = stft(signal, SR, frame_sec, hop_sec)

    center_pad      = frame_len // 2
    padded_len      = len(signal) + 2 * center_pad
    expected_bins   = frame_len // 2 + 1
    expected_frames = 1 + (padded_len - frame_len + hop_len - 1) // hop_len

    assert S.shape[1] == expected_bins,   f"bins mismatch: {S.shape[1]} vs {expected_bins}"
    assert S.shape[0] == expected_frames, f"frames mismatch: {S.shape[0]} vs {expected_frames}"
    print(f"[PASS] Shape  frames={S.shape[0]}, bins={S.shape[1]}")


# ── Test 3: spectral peak at correct bin ─────────────────────────────────────
def test_spectral_peak():
    freq   = 440.0
    signal = make_sine(freq, 1.0, SR)
    S, frame_len, hop_len = stft(signal, SR)

    magnitude, _ = magnitude_phase(S)
    mean_mag     = magnitude.mean(axis=0)   # average over time

    # Frequency resolution
    freq_res  = SR / frame_len
    peak_bin  = np.argmax(mean_mag)
    peak_freq = peak_bin * freq_res

    assert abs(peak_freq - freq) < freq_res, (
        f"Peak at {peak_freq:.1f} Hz, expected ~{freq} Hz (res={freq_res:.1f} Hz)"
    )
    print(f"[PASS] Spectral peak  peak_bin={peak_bin}, peak_freq={peak_freq:.1f} Hz")


# ── Test 4: magnitude+phase round-trip ───────────────────────────────────────
def test_magnitude_phase_roundtrip():
    signal = make_sine(440, 0.5, SR)
    S, frame_len, hop_len = stft(signal, SR)

    mag, phase = magnitude_phase(S)
    S2         = reconstruct_from_magnitude(mag, phase)

    err = np.max(np.abs(S - S2))
    assert err < 1e-10, f"mag/phase round-trip error: {err:.2e}"
    print(f"[PASS] Magnitude+phase round-trip  err={err:.2e}")


if __name__ == "__main__":
    test_shape()
    test_perfect_reconstruction()
    test_spectral_peak()
    test_magnitude_phase_roundtrip()
    print("\nAll tests passed.")
