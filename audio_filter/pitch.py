"""
Pitch (F0) detector — built from first principles using NumPy.

Algorithm: autocorrelation method with parabolic interpolation.

Per-frame pipeline:
  window frame → autocorrelation → normalize by R[0]
  → search peak in [tau_min, tau_max]
  → parabolic interpolation for sub-sample lag accuracy
  → F0 = sample_rate / tau*
  → voiced if peak height > voiced_threshold

Jitter (period perturbation quotient):
  PPQ = mean(|T[n+1] - T[n]|) / mean(T)   over voiced frames only

Typical F0 ranges:
  Male speech   :  80 – 180 Hz
  Female speech : 160 – 300 Hz
  General range :  60 – 400 Hz
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal framing  (mirrors stft.py and lpc.py conventions)
# ---------------------------------------------------------------------------

def _make_frames(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """
    Center-padded frames WITHOUT windowing, shape (n_frames, frame_len).

    Pitch detection uses the autocorrelation of the raw (rectangular-windowed)
    frame.  Applying a Hann window here would reduce the ACF magnitude to ~0.25
    even for a clean periodic signal, making the voiced threshold unreliable.
    Windowing belongs to FFT-based analysis (STFT/MFCC), not to ACF pitch.
    """
    center_pad = frame_len // 2
    sig = np.concatenate([np.zeros(center_pad), signal, np.zeros(center_pad)])

    n_frames  = 1 + (len(sig) - frame_len + hop_len - 1) // hop_len
    pad_right = (n_frames - 1) * hop_len + frame_len - len(sig)
    if pad_right > 0:
        sig = np.concatenate([sig, np.zeros(pad_right)])

    indices = (
        np.arange(frame_len)[np.newaxis, :]
        + np.arange(n_frames)[:, np.newaxis] * hop_len
    )
    return sig[indices]   # rectangular window — no Hann applied


# ---------------------------------------------------------------------------
# Autocorrelation of a single frame
# ---------------------------------------------------------------------------

def _autocorr(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation r[0..max_lag].

    r[k] = sum_{n=0}^{N-1-k} frame[n]*frame[n+k]  /  sum_{n=0}^{N-1} frame[n]^2

    r[0] = 1 by definition.  Values in [-1, 1].
    """
    R0 = np.dot(frame, frame)
    if R0 < 1e-10:
        return np.zeros(max_lag + 1)

    r = np.empty(max_lag + 1)
    r[0] = 1.0
    for k in range(1, max_lag + 1):
        r[k] = np.dot(frame[:len(frame) - k], frame[k:]) / R0
    return r


# ---------------------------------------------------------------------------
# Parabolic interpolation around a peak
# ---------------------------------------------------------------------------

def _parabolic_peak(r: np.ndarray, peak_idx: int) -> float:
    """
    Refine peak location to sub-sample precision by fitting a parabola
    through (peak_idx-1, peak_idx, peak_idx+1).

    Returns the fractional index of the interpolated peak.
    """
    if peak_idx <= 0 or peak_idx >= len(r) - 1:
        return float(peak_idx)

    y0, y1, y2 = r[peak_idx - 1], r[peak_idx], r[peak_idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(peak_idx)

    return peak_idx - 0.5 * (y2 - y0) / denom


# ---------------------------------------------------------------------------
# Single-frame pitch estimator
# ---------------------------------------------------------------------------

def _pitch_frame(
    frame: np.ndarray,
    sample_rate: int,
    tau_min: int,
    tau_max: int,
    voiced_threshold: float,
) -> tuple[float, float]:
    """
    Estimate F0 from one windowed frame.

    Returns
    -------
    f0         : Hz, or nan if unvoiced
    confidence : normalized autocorrelation peak height in [0, 1]
    """
    r = _autocorr(frame, tau_max)

    # Search for the highest peak in the valid lag window
    segment    = r[tau_min : tau_max + 1]
    local_peak = int(np.argmax(segment))
    global_idx = local_peak + tau_min
    confidence = float(r[global_idx])

    if confidence < voiced_threshold:
        return np.nan, confidence

    # Refine with parabolic interpolation
    tau_exact = _parabolic_peak(r, global_idx)
    if tau_exact <= 0:
        return np.nan, confidence

    f0 = sample_rate / tau_exact
    return f0, confidence


# ---------------------------------------------------------------------------
# Per-signal pitch track
# ---------------------------------------------------------------------------

def pitch(
    signal: np.ndarray,
    sample_rate: int,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    voiced_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate F0 for every frame of a mono signal.

    Parameters
    ----------
    signal           : 1-D float array
    sample_rate      : samples per second
    f0_min           : minimum detectable F0 in Hz  (default 60 Hz)
    f0_max           : maximum detectable F0 in Hz  (default 400 Hz)
    frame_sec        : frame length in seconds
    hop_sec          : hop size in seconds
    voiced_threshold : normalized autocorr peak to call a frame voiced
                       (0.45 is a conservative starting point)

    Returns
    -------
    f0          : (n_frames,) array — Hz for voiced frames, nan for unvoiced
    confidence  : (n_frames,) array — autocorrelation peak height in [0, 1]
    """
    signal    = np.asarray(signal, dtype=np.float64)
    frame_len = int(round(frame_sec * sample_rate))
    hop_len   = int(round(hop_sec   * sample_rate))

    tau_min = max(1, int(np.floor(sample_rate / f0_max)))
    tau_max = min(frame_len - 1, int(np.ceil(sample_rate / f0_min)))

    frames     = _make_frames(signal, frame_len, hop_len)   # (n_frames, frame_len)
    n_frames   = frames.shape[0]
    f0_track   = np.full(n_frames, np.nan)
    conf_track = np.zeros(n_frames)

    for i in range(n_frames):
        f0_track[i], conf_track[i] = _pitch_frame(
            frames[i], sample_rate, tau_min, tau_max, voiced_threshold
        )

    return f0_track, conf_track


# ---------------------------------------------------------------------------
# Jitter  (period perturbation quotient)
# ---------------------------------------------------------------------------

def jitter(f0_track: np.ndarray, sample_rate: int) -> float:
    """
    Compute the Period Perturbation Quotient (PPQ) from a pitch track.

    PPQ = mean(|T[n+1] - T[n]|) / mean(T)

    Only consecutive voiced frames are used.  Returns nan if fewer than
    two voiced frames exist.

    Parameters
    ----------
    f0_track    : (n_frames,) array from pitch(), nan = unvoiced
    sample_rate : samples per second

    Returns
    -------
    ppq : float — jitter ratio (dimensionless), typically 0.005 – 0.02 for
          healthy voice, > 0.03 for dysphonia indicators
    """
    voiced = f0_track[~np.isnan(f0_track)]
    if len(voiced) < 2:
        return np.nan

    periods = sample_rate / voiced
    return float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
