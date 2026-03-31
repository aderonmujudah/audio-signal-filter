"""
Audio Filter — FastAPI backend.

Endpoints
─────────
GET  /health          liveness check
POST /analyze         feature extraction (pitch, ZCR, energy, VAD, MFCCs, formants)
POST /separate        source separation  (ICA for multi-channel, NMF for mono)
POST /diarize         speaker diarization via GMM + BIC on MFCC features

Audio I/O
─────────
Input  : WAV file upload (mono or stereo, any standard PCM / float format)
Output : JSON; separated audio is base64-encoded 16-bit PCM WAV
"""

import io
import base64

import numpy as np
import scipy.io.wavfile as _wavfile
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_filter.ica         import ica
from audio_filter.lpc         import formants as lpc_formants
from audio_filter.mfcc        import mfcc, delta
from audio_filter.nmf         import nmf, griffin_lim
from audio_filter.pitch       import pitch
from audio_filter.zcr_energy  import (
    frame_energy, log_energy, speech_activity, zcr,
    SILENCE, VOICED, UNVOICED,
)
from audio_filter.gmm         import select_n_components, gmm_predict, gmm_fit

_EPS = 1e-10

app = FastAPI(title="Audio Filter API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """
    Parse WAV bytes → float64 signal array and sample rate.

    Returns
    -------
    signal : (n_samples,) for mono, (n_samples, n_channels) for multi-channel
    sr     : int
    """
    buf = io.BytesIO(data)
    try:
        sr, signal = _wavfile.read(buf)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read WAV: {exc}")

    # Normalize to float64 in [-1, 1]
    if signal.dtype == np.uint8:
        signal = signal.astype(np.float64) / 128.0 - 1.0
    elif signal.dtype == np.int16:
        signal = signal.astype(np.float64) / 32768.0
    elif signal.dtype == np.int32:
        signal = signal.astype(np.float64) / 2 ** 31
    elif signal.dtype in (np.float32,):
        signal = signal.astype(np.float64)
    elif signal.dtype == np.float64:
        pass
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported WAV dtype: {signal.dtype}")

    return signal, int(sr)


def _to_mono(signal: np.ndarray) -> np.ndarray:
    """Return first channel of a (n_samples, n_ch) array, or pass through 1-D."""
    if signal.ndim == 2:
        return signal[:, 0]
    return signal


def _encode_wav(signal: np.ndarray, sr: int) -> str:
    """float64 signal → base64-encoded 16-bit PCM WAV string."""
    sig16 = np.clip(signal, -1.0, 1.0)
    sig16 = (sig16 * 32767).astype(np.int16)
    buf = io.BytesIO()
    _wavfile.write(buf, sr, sig16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _nan_to_none(x):
    """Scalar float → None if NaN, else Python float."""
    return None if (x is not None and np.isnan(x)) else (float(x) if x is not None else None)


def _arr_to_list(arr: np.ndarray) -> list:
    """1-D numpy array → Python list; NaN → None."""
    return [None if np.isnan(v) else float(v) for v in arr]


def _arr2d_to_list(arr: np.ndarray) -> list:
    """2-D numpy array → list-of-lists; NaN → None."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    n_mfcc: int = Form(13),
    f0_min: float = Form(60.0),
    f0_max: float = Form(400.0),
    n_formants: int = Form(4),
):
    """
    Extract features from a mono WAV file.

    Returns
    -------
    JSON with:
      info      : {sample_rate, duration, n_channels}
      pitch     : {f0, confidence, frame_times}   (Hz; null = unvoiced)
      energy    : {rms, log_rms, frame_times}
      zcr       : {zcr, frame_times}
      vad       : {labels, frame_times}            (0=silence, 1=voiced, 2=unvoiced)
      mfcc      : {coeffs, delta1, delta2, frame_times}   shape (n_frames, n_mfcc)
      formants  : {freqs, bws, frame_times}        shape (n_frames, n_formants)
    """
    raw = await file.read()
    signal, sr = _read_wav(raw)
    n_ch = signal.shape[1] if signal.ndim == 2 else 1
    mono = _to_mono(signal)
    duration = len(mono) / sr

    frame_sec = 0.025
    hop_sec   = 0.010

    # Frame times (centre of each frame, approximate)
    def _frame_times(n_frames):
        return [round(i * hop_sec + frame_sec / 2, 6) for i in range(n_frames)]

    # ── pitch ────────────────────────────────────────────────────────────────
    f0_track, conf_track = pitch(mono, sr, f0_min=f0_min, f0_max=f0_max,
                                  frame_sec=frame_sec, hop_sec=hop_sec)

    # ── ZCR / energy / VAD ───────────────────────────────────────────────────
    zcr_track = zcr(mono, sr, frame_sec=frame_sec, hop_sec=hop_sec)
    rms_track = frame_energy(mono, sr, frame_sec=frame_sec, hop_sec=hop_sec)
    log_track  = log_energy(mono, sr, frame_sec=frame_sec, hop_sec=hop_sec)
    vad_labels = speech_activity(zcr_track, rms_track)

    n_frames = len(f0_track)
    ft = _frame_times(n_frames)

    # ── MFCCs + deltas ───────────────────────────────────────────────────────
    coeffs   = mfcc(mono, sr, n_mfcc=n_mfcc, frame_sec=frame_sec, hop_sec=hop_sec)
    d1       = delta(coeffs, width=9)
    d2       = delta(d1,     width=9)
    mfcc_ft  = _frame_times(len(coeffs))

    # ── formants ─────────────────────────────────────────────────────────────
    try:
        freqs, bws = lpc_formants(mono, sr, n_formants=n_formants,
                                  frame_sec=frame_sec, hop_sec=hop_sec)
    except Exception:
        n_fr = len(coeffs)
        freqs = np.full((n_fr, n_formants), np.nan)
        bws   = np.full((n_fr, n_formants), np.nan)

    return {
        "info": {
            "sample_rate": sr,
            "duration":    round(duration, 4),
            "n_channels":  n_ch,
        },
        "pitch": {
            "f0":          _arr_to_list(f0_track),
            "confidence":  _arr_to_list(conf_track),
            "frame_times": ft,
        },
        "energy": {
            "rms":         _arr_to_list(rms_track),
            "log_rms":     _arr_to_list(log_track),
            "frame_times": _frame_times(len(rms_track)),
        },
        "zcr": {
            "zcr":         _arr_to_list(zcr_track),
            "frame_times": _frame_times(len(zcr_track)),
        },
        "vad": {
            "labels":      vad_labels.tolist(),
            "frame_times": _frame_times(len(vad_labels)),
            "legend":      {"0": "silence", "1": "voiced", "2": "unvoiced"},
        },
        "mfcc": {
            "coeffs":      _arr2d_to_list(coeffs),
            "delta1":      _arr2d_to_list(d1),
            "delta2":      _arr2d_to_list(d2),
            "frame_times": mfcc_ft,
        },
        "formants": {
            "freqs":       _arr2d_to_list(freqs),
            "bws":         _arr2d_to_list(bws),
            "frame_times": _frame_times(len(freqs)),
        },
    }


@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    n_sources: int = Form(2),
    method: str = Form("auto"),
    gl_iter: int = Form(64),
):
    """
    Separate audio into independent sources.

    Parameters
    ----------
    n_sources : number of sources to extract (NMF only; ICA uses channel count)
    method    : 'auto' | 'ica' | 'nmf'
                auto → ICA if ≥ 2 channels, NMF if mono
    gl_iter   : Griffin-Lim iterations for NMF reconstruction

    Returns
    -------
    JSON with:
      method   : 'ica' or 'nmf'
      sources  : list of {index, wav_b64}   (16-bit PCM WAV, base64-encoded)
    """
    raw = await file.read()
    signal, sr = _read_wav(raw)
    n_ch = signal.shape[1] if signal.ndim == 2 else 1

    # Resolve method
    if method == "auto":
        chosen = "ica" if n_ch >= 2 else "nmf"
    elif method in ("ica", "nmf"):
        chosen = method
    else:
        raise HTTPException(status_code=422, detail=f"method must be 'auto', 'ica', or 'nmf'")

    if chosen == "ica":
        if n_ch < 2:
            raise HTTPException(status_code=422,
                detail="ICA requires a multi-channel (stereo+) WAV file")
        # Transpose to (n_channels, n_samples)
        X = signal.T.astype(np.float64)          # (n_ch, n_samples)
        S, _ = ica(X)                            # (n_ch, n_samples)
        sources = []
        for i, s in enumerate(S):
            s_norm = s / (np.max(np.abs(s)) + _EPS) * 0.9
            sources.append({"index": i, "wav_b64": _encode_wav(s_norm, sr)})
        return {"method": "ica", "sources": sources}

    else:  # nmf
        mono = _to_mono(signal)
        if n_sources < 1:
            raise HTTPException(status_code=422, detail="n_sources must be ≥ 1")

        S_stft, frame_len, hop_len = _stft_for_nmf(mono, sr)
        V = np.abs(S_stft).T                     # (n_bins, n_frames)

        W, H = nmf(V, n_components=n_sources)

        WH = W @ H + _EPS
        sources = []
        for i in range(n_sources):
            # Soft Wiener mask: preserves phase-consistent magnitude estimate
            mask  = (W[:, i:i+1] * H[i:i+1, :]) / WH   # (n_bins, n_frames)
            mag_i = mask * V
            audio = griffin_lim(mag_i, frame_len, hop_len, sr, n_iter=gl_iter)
            audio = audio / (np.max(np.abs(audio)) + _EPS) * 0.9
            sources.append({"index": i, "wav_b64": _encode_wav(audio, sr)})

        return {"method": "nmf", "sources": sources}


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    max_speakers: int = Form(5),
    n_mfcc: int = Form(13),
):
    """
    Speaker diarization via GMM clustering on MFCC + delta features.

    BIC selects the number of speakers automatically (up to max_speakers).

    Returns
    -------
    JSON with:
      n_speakers   : int
      frame_labels : list of int  (speaker index per MFCC frame)
      frame_times  : list of float  (seconds)
      segments     : list of {start, end, speaker}
    """
    raw = await file.read()
    signal, sr = _read_wav(raw)
    mono = _to_mono(signal)

    if max_speakers < 1:
        raise HTTPException(status_code=422, detail="max_speakers must be ≥ 1")

    frame_sec = 0.025
    hop_sec   = 0.010

    # ── feature extraction: MFCC + Δ + ΔΔ ──────────────────────────────────
    coeffs = mfcc(mono, sr, n_mfcc=n_mfcc, frame_sec=frame_sec, hop_sec=hop_sec)
    d1     = delta(coeffs, width=9)
    d2     = delta(d1,     width=9)
    feats  = np.hstack([coeffs, d1, d2])     # (n_frames, 3*n_mfcc)

    # ── normalize per feature ────────────────────────────────────────────────
    mu    = feats.mean(axis=0)
    sigma = feats.std(axis=0) + _EPS
    feats = (feats - mu) / sigma

    n_frames = len(feats)
    if n_frames < max_speakers:
        max_speakers = max(1, n_frames)

    # ── BIC model selection ──────────────────────────────────────────────────
    if max_speakers == 1:
        model   = gmm_fit(feats, 1, covariance_type="diag", n_init=3)
        best_k  = 1
    else:
        best_k, model, _ = select_n_components(
            feats,
            range(1, max_speakers + 1),
            covariance_type="diag",
            max_iter=100,
            tol=1e-4,
            n_init=3,
        )

    labels = gmm_predict(feats, model)

    # ── frame times ──────────────────────────────────────────────────────────
    frame_times = [round(i * hop_sec + frame_sec / 2, 6) for i in range(n_frames)]

    # ── merge consecutive same-speaker frames into segments ──────────────────
    segments = []
    if n_frames > 0:
        seg_start = 0
        for i in range(1, n_frames):
            if labels[i] != labels[i - 1]:
                segments.append({
                    "start":   frame_times[seg_start],
                    "end":     frame_times[i - 1],
                    "speaker": int(labels[seg_start]),
                })
                seg_start = i
        segments.append({
            "start":   frame_times[seg_start],
            "end":     frame_times[n_frames - 1],
            "speaker": int(labels[seg_start]),
        })

    return {
        "n_speakers":   int(best_k),
        "frame_labels": labels.tolist(),
        "frame_times":  frame_times,
        "segments":     segments,
    }


# ---------------------------------------------------------------------------
# Internal helper — STFT params for NMF (shared convention)
# ---------------------------------------------------------------------------

def _stft_for_nmf(mono: np.ndarray, sr: int):
    """Run STFT with standard speech params; return (S, frame_len, hop_len)."""
    from audio_filter.stft import stft as _stft
    return _stft(mono, sr, frame_sec=0.025, hop_sec=0.010)


# ---------------------------------------------------------------------------
# Entry point  (uvicorn app:app --reload)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
