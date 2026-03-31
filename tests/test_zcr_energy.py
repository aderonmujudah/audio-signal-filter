"""
Tests for ZCR and short-term energy.

Run with:  python tests/test_zcr_energy.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_filter.zcr_energy import (
    zcr, frame_energy, log_energy, speech_activity,
    SILENCE, VOICED, UNVOICED,
)

SR = 16000


def make_sine(freq, duration=1.0, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


def make_voiced_speech(f0=120, duration=1.0, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 10))
    return sig / np.max(np.abs(sig))


# ── ZCR: output in [0, 1] ────────────────────────────────────────────────────
def test_zcr_range():
    signal   = make_sine(440)
    z        = zcr(signal, SR)
    assert np.all(z >= 0) and np.all(z <= 1), f"ZCR out of [0,1]: min={z.min():.4f} max={z.max():.4f}"
    print(f"[PASS] ZCR range  min={z.min():.4f}  max={z.max():.4f}")


# ── ZCR: silence → 0 crossings ───────────────────────────────────────────────
def test_zcr_silence():
    silence = np.zeros(SR)
    z       = zcr(silence, SR)
    assert np.all(z == 0.0), "Silence should have ZCR == 0"
    print("[PASS] ZCR silence -> 0")


# ── ZCR: high-frequency signal has higher ZCR than low-frequency ─────────────
def test_zcr_ordering():
    lo = make_sine(100)
    hi = make_sine(4000)
    z_lo = zcr(lo, SR).mean()
    z_hi = zcr(hi, SR).mean()
    assert z_hi > z_lo, f"Expected ZCR(4kHz) > ZCR(100Hz), got {z_hi:.4f} vs {z_lo:.4f}"
    print(f"[PASS] ZCR ordering  ZCR(100Hz)={z_lo:.4f}  ZCR(4kHz)={z_hi:.4f}")


# ── ZCR: known frequency matches theory ──────────────────────────────────────
def test_zcr_known_freq():
    # A sine at frequency f crosses zero 2*f times per second.
    # ZCR per frame = 2*f * frame_sec  (approximately)
    freq      = 440.0
    frame_sec = 0.025
    signal    = make_sine(freq)
    z         = zcr(signal, SR, frame_sec=frame_sec)

    # crossings per frame ≈ 2*f*frame_sec, normalized by (frame_len - 1)
    frame_len = int(round(frame_sec * SR))
    expected  = (2 * freq * frame_sec) / (frame_len - 1)
    mean_z    = z[5:-5].mean()              # skip edge frames
    assert abs(mean_z - expected) < 0.005, (
        f"ZCR={mean_z:.4f}  expected~{expected:.4f}"
    )
    print(f"[PASS] ZCR theory  got={mean_z:.4f}  expected={expected:.4f}")


# ── Energy: silence → zero ────────────────────────────────────────────────────
def test_energy_silence():
    E = frame_energy(np.zeros(SR), SR)
    assert np.all(E == 0.0)
    print("[PASS] energy(silence) == 0")


# ── Energy: scales with amplitude squared ────────────────────────────────────
def test_energy_amplitude_scaling():
    signal = make_sine(440)
    E1 = frame_energy(signal,      SR).mean()
    E2 = frame_energy(signal * 2,  SR).mean()
    E4 = frame_energy(signal * 4,  SR).mean()
    assert abs(E2 / E1 - 4.0) < 0.01, f"E(2x) / E(1x) = {E2/E1:.4f}, expected 4.0"
    assert abs(E4 / E1 - 16.) < 0.01, f"E(4x) / E(1x) = {E4/E1:.4f}, expected 16.0"
    print(f"[PASS] energy amplitude scaling  E(2x)/E(x)={E2/E1:.4f}  E(4x)/E(x)={E4/E1:.4f}")


# ── Log energy: monotone with amplitude ──────────────────────────────────────
def test_log_energy_monotone():
    signal = make_sine(440)
    lE1 = log_energy(signal,      SR).mean()
    lE2 = log_energy(signal * 2,  SR).mean()
    assert lE2 > lE1, f"log_energy(2x) should be > log_energy(x)"
    diff = lE2 - lE1
    assert abs(diff - np.log(4)) < 0.01, f"Expected diff={np.log(4):.4f}, got {diff:.4f}"
    print(f"[PASS] log energy  lE(2x)-lE(x)={diff:.4f}  expected={np.log(4):.4f}")


# ── speech_activity labels ────────────────────────────────────────────────────
def test_speech_activity():
    voiced  = make_voiced_speech(f0=120)
    silence = np.zeros(SR)

    # Concatenate: 0.5s voiced, 0.5s silence
    mix     = np.concatenate([voiced[:SR//2], silence[:SR//2]])
    z       = zcr(mix, SR)
    E       = frame_energy(mix, SR)
    labels  = speech_activity(z, E)

    # At least some frames should be VOICED and some SILENCE
    assert np.any(labels == VOICED),   "No VOICED frames detected"
    assert np.any(labels == SILENCE),  "No SILENCE frames detected"

    # Voiced section (first half) should be mostly not-silence
    mid = len(labels) // 2
    voiced_section_active = np.mean(labels[:mid] != SILENCE)
    assert voiced_section_active > 0.5, (
        f"Voiced section has too few active frames: {voiced_section_active:.2f}"
    )
    # Silence section (second half) should be mostly silence
    silence_section = np.mean(labels[mid:] == SILENCE)
    assert silence_section > 0.8, (
        f"Silence section has too few silence frames: {silence_section:.2f}"
    )

    print(f"[PASS] speech_activity  voiced_section_active={voiced_section_active:.2f}  "
          f"silence_section={silence_section:.2f}")


if __name__ == "__main__":
    test_zcr_range()
    test_zcr_silence()
    test_zcr_ordering()
    test_zcr_known_freq()
    test_energy_silence()
    test_energy_amplitude_scaling()
    test_log_energy_monotone()
    test_speech_activity()
    print("\nAll tests passed.")
