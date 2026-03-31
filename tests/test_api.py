"""
Integration tests for the FastAPI backend.

Run with:  python tests/test_api.py

Uses FastAPI's TestClient (no server needed — httpx in-process).
"""

import io
import sys
import os
import base64
import numpy as np
import scipy.io.wavfile as _wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)
SR = 16000


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(signal: np.ndarray, sr: int = SR) -> bytes:
    """float64 signal → in-memory WAV bytes (int16 PCM)."""
    sig16 = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    _wavfile.write(buf, sr, sig16)
    return buf.getvalue()


def _make_stereo_wav_bytes(s1: np.ndarray, s2: np.ndarray, sr: int = SR) -> bytes:
    """Two float64 signals → stereo WAV bytes."""
    stereo = np.column_stack([
        (np.clip(s1, -1.0, 1.0) * 32767).astype(np.int16),
        (np.clip(s2, -1.0, 1.0) * 32767).astype(np.int16),
    ])
    buf = io.BytesIO()
    _wavfile.write(buf, sr, stereo)
    return buf.getvalue()


def _sine(freq=440, duration=1.0, sr=SR):
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t)


# ── health check ──────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[PASS] GET /health")


# ── /analyze: response structure ──────────────────────────────────────────────
def test_analyze_structure():
    wav = _make_wav_bytes(_sine())
    r   = client.post("/analyze", files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 200, f"status={r.status_code}  body={r.text[:200]}"
    j   = r.json()
    for key in ("info", "pitch", "energy", "zcr", "vad", "mfcc", "formants"):
        assert key in j, f"Missing key: {key}"
    assert j["info"]["sample_rate"] == SR
    assert j["info"]["n_channels"]  == 1
    assert j["info"]["duration"]    > 0.9
    print(f"[PASS] /analyze structure  sr={j['info']['sample_rate']}  "
          f"dur={j['info']['duration']:.2f}s")


# ── /analyze: frame counts are consistent ─────────────────────────────────────
def test_analyze_frame_counts():
    wav = _make_wav_bytes(_sine(duration=0.5))
    r   = client.post("/analyze", files={"file": ("test.wav", wav, "audio/wav")})
    j   = r.json()
    n_pitch  = len(j["pitch"]["f0"])
    n_mfcc   = len(j["mfcc"]["coeffs"])
    n_energy = len(j["energy"]["rms"])
    # All feature streams should have the same number of frames
    assert n_pitch == n_energy, f"pitch frames {n_pitch} != energy frames {n_energy}"
    assert n_mfcc  == n_energy, f"mfcc frames {n_mfcc} != energy frames {n_energy}"
    print(f"[PASS] /analyze frame counts consistent  n_frames={n_pitch}")


# ── /analyze: MFCC shape ──────────────────────────────────────────────────────
def test_analyze_mfcc_shape():
    wav  = _make_wav_bytes(_sine())
    r    = client.post("/analyze",
                       data={"n_mfcc": 13},
                       files={"file": ("test.wav", wav, "audio/wav")})
    j    = r.json()
    rows = j["mfcc"]["coeffs"]
    assert len(rows) > 0
    assert len(rows[0]) == 13, f"Expected 13 MFCC coeffs, got {len(rows[0])}"
    print(f"[PASS] /analyze MFCC shape  ({len(rows)}, {len(rows[0])})")


# ── /analyze: voiced sine has non-null f0 ─────────────────────────────────────
def test_analyze_pitch_detection():
    wav = _make_wav_bytes(_sine(freq=220, duration=1.0))
    r   = client.post("/analyze", files={"file": ("test.wav", wav, "audio/wav")})
    j   = r.json()
    f0  = [x for x in j["pitch"]["f0"] if x is not None]
    assert len(f0) > 0, "No voiced frames detected for 220 Hz sine"
    mean_f0 = sum(f0) / len(f0)
    assert 200 < mean_f0 < 240, f"Mean F0={mean_f0:.1f} Hz, expected ~220 Hz"
    print(f"[PASS] /analyze pitch  mean_f0={mean_f0:.1f} Hz  voiced_frames={len(f0)}")


# ── /separate NMF: returns n_sources audio blobs ──────────────────────────────
def test_separate_nmf_shape():
    wav = _make_wav_bytes(_sine() + 0.5 * _sine(880))
    r   = client.post("/separate",
                      data={"n_sources": 2, "method": "nmf", "gl_iter": 8},
                      files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 200, f"{r.status_code}  {r.text[:200]}"
    j   = r.json()
    assert j["method"] == "nmf"
    assert len(j["sources"]) == 2
    # Each source must be decodeable WAV
    for src in j["sources"]:
        raw = base64.b64decode(src["wav_b64"])
        sr2, sig = _wavfile.read(io.BytesIO(raw))
        assert sr2 == SR
        assert len(sig) > 0
    print(f"[PASS] /separate NMF  n_sources={len(j['sources'])}")


# ── /separate ICA: returns n_channels sources ─────────────────────────────────
def test_separate_ica_shape():
    s1  = _sine(440)
    s2  = _sine(880)
    wav = _make_stereo_wav_bytes(s1, s2)
    r   = client.post("/separate",
                      data={"method": "ica"},
                      files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 200, f"{r.status_code}  {r.text[:200]}"
    j   = r.json()
    assert j["method"] == "ica"
    assert len(j["sources"]) == 2
    for src in j["sources"]:
        raw = base64.b64decode(src["wav_b64"])
        sr2, sig = _wavfile.read(io.BytesIO(raw))
        assert sr2 == SR
        assert len(sig) > 0
    print(f"[PASS] /separate ICA  n_sources={len(j['sources'])}")


# ── /separate: ICA on mono returns 422 ───────────────────────────────────────
def test_separate_ica_mono_error():
    wav = _make_wav_bytes(_sine())
    r   = client.post("/separate",
                      data={"method": "ica"},
                      files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("[PASS] /separate ICA mono -> 422")


# ── /diarize: response structure ─────────────────────────────────────────────
def test_diarize_structure():
    # Two seconds of alternating content
    t  = np.arange(SR * 2) / SR
    s  = np.sin(2 * np.pi * 120 * t) * (t < 1) + np.sin(2 * np.pi * 300 * t) * (t >= 1)
    wav = _make_wav_bytes(s)
    r   = client.post("/diarize",
                      data={"max_speakers": 3},
                      files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 200, f"{r.status_code}  {r.text[:200]}"
    j   = r.json()
    for key in ("n_speakers", "frame_labels", "frame_times", "segments"):
        assert key in j, f"Missing key: {key}"
    assert 1 <= j["n_speakers"] <= 3
    assert len(j["frame_labels"]) == len(j["frame_times"])
    assert len(j["segments"]) >= 1
    # Segment boundaries must be non-decreasing
    for seg in j["segments"]:
        assert seg["end"] >= seg["start"]
    print(f"[PASS] /diarize structure  n_speakers={j['n_speakers']}  "
          f"n_segments={len(j['segments'])}")


# ── /diarize: labels are in [0, n_speakers) ──────────────────────────────────
def test_diarize_label_range():
    wav = _make_wav_bytes(_sine(duration=1.0))
    r   = client.post("/diarize",
                      data={"max_speakers": 4},
                      files={"file": ("test.wav", wav, "audio/wav")})
    j   = r.json()
    k   = j["n_speakers"]
    labels = j["frame_labels"]
    assert all(0 <= lbl < k for lbl in labels), (
        f"Labels out of [0, {k}): {set(labels)}"
    )
    print(f"[PASS] /diarize label range  k={k}  label_set={sorted(set(labels))}")


if __name__ == "__main__":
    test_health()
    test_analyze_structure()
    test_analyze_frame_counts()
    test_analyze_mfcc_shape()
    test_analyze_pitch_detection()
    test_separate_nmf_shape()
    test_separate_ica_shape()
    test_separate_ica_mono_error()
    test_diarize_structure()
    test_diarize_label_range()
    print("\nAll tests passed.")
