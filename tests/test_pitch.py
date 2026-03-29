"""
Tests for the pitch (F0) detector.

Run with:  python tests/test_pitch.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.pitch import pitch, jitter

SR = 16000


def make_sine(freq, duration=1.0, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def make_voiced(f0=120, duration=1.0, sr=SR):
    """Harmonic stack — a clean periodic signal the detector should track."""
    t = np.arange(int(duration * sr)) / sr
    signal = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 16))
    return signal / np.max(np.abs(signal))


# ── known pitch detected correctly ───────────────────────────────────────────
def test_known_pitch():
    for target in [100, 150, 200, 250]:
        signal = make_voiced(f0=target)
        f0, conf = pitch(signal, SR)

        voiced    = f0[~np.isnan(f0)]
        assert len(voiced) > 0, f"No voiced frames for f0={target}"
        mean_f0   = voiced.mean()
        tolerance = target * 0.05          # within 5 %
        assert abs(mean_f0 - target) < tolerance, (
            f"f0={target}: mean={mean_f0:.1f}  tol={tolerance:.1f}"
        )
        print(f"[PASS] f0={target} Hz  detected={mean_f0:.1f} Hz  "
              f"voiced_frames={len(voiced)}/{len(f0)}")


# ── silence → all unvoiced ────────────────────────────────────────────────────
def test_silence_unvoiced():
    silence = np.zeros(SR)
    f0, conf = pitch(silence, SR)
    assert np.all(np.isnan(f0)), "Silence should produce all-NaN pitch track"
    assert np.all(conf == 0.0)
    print("[PASS] silence -> all unvoiced")


# ── white noise → mostly unvoiced ────────────────────────────────────────────
def test_noise_mostly_unvoiced():
    rng   = np.random.default_rng(0)
    noise = rng.standard_normal(SR)
    f0, conf = pitch(noise, SR, voiced_threshold=0.45)

    voiced_ratio = np.mean(~np.isnan(f0))
    assert voiced_ratio < 0.25, f"Too many voiced frames in noise: {voiced_ratio:.2f}"
    print(f"[PASS] noise voiced_ratio={voiced_ratio:.3f}")


# ── output shape consistency ──────────────────────────────────────────────────
def test_output_shape():
    signal   = make_voiced(f0=150)
    f0, conf = pitch(signal, SR)

    assert f0.shape == conf.shape
    assert f0.ndim == 1
    assert len(f0) > 0
    print(f"[PASS] output shape  n_frames={len(f0)}")


# ── jitter near zero for a clean periodic signal ─────────────────────────────
def test_jitter_clean():
    signal    = make_voiced(f0=150)
    f0, _     = pitch(signal, SR)
    ppq       = jitter(f0, SR)

    assert not np.isnan(ppq),   "jitter returned NaN for voiced signal"
    assert ppq < 0.05,          f"jitter too large for clean periodic signal: {ppq:.4f}"
    print(f"[PASS] jitter={ppq:.6f}  (clean periodic signal)")


# ── jitter is NaN for silence (no voiced frames) ─────────────────────────────
def test_jitter_silence():
    silence = np.zeros(SR)
    f0, _   = pitch(silence, SR)
    ppq     = jitter(f0, SR)
    assert np.isnan(ppq), "jitter of all-unvoiced signal should be NaN"
    print("[PASS] jitter(silence) -> NaN")


if __name__ == "__main__":
    test_known_pitch()
    test_silence_unvoiced()
    test_noise_mostly_unvoiced()
    test_output_shape()
    test_jitter_clean()
    test_jitter_silence()
    print("\nAll tests passed.")
