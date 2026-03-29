"""
Zero Crossing Rate (ZCR) and short-term energy — from first principles.

Both are computed per frame using the same center-pad / rectangular-window
framing convention as the pitch module.

ZCR:
    zcr[n] = #{k : x[k]*x[k-1] < 0} / (frame_len - 1)
    High ZCR → unvoiced / fricative / noise
    Low  ZCR → voiced speech or silence

Short-term energy:
    E[n]   = (1/N) * sum_{k=0}^{N-1} x[k]^2
    log_E  = log(max(E, floor))

Together they give a simple 3-class activity label:
    silence   : E  < energy_threshold
    voiced    : E >= energy_threshold  AND  zcr < zcr_threshold
    unvoiced  : E >= energy_threshold  AND  zcr >= zcr_threshold
"""

import numpy as np

# Activity labels (integers so they can be stored in an int array)
SILENCE   = 0
VOICED    = 1
UNVOICED  = 2


# ---------------------------------------------------------------------------
# Shared framing  (rectangular window — same rationale as pitch.py)
# ---------------------------------------------------------------------------

def _make_frames(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Center-padded rectangular frames, shape (n_frames, frame_len)."""
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
    return sig[indices]   # (n_frames, frame_len)


# ---------------------------------------------------------------------------
# Zero Crossing Rate
# ---------------------------------------------------------------------------

def zcr(
    signal: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
) -> np.ndarray:
    """
    Zero Crossing Rate per frame.

    ZCR[n] = #{k in [1,N-1] : frame[k]*frame[k-1] < 0}  /  (N - 1)

    A sign change is counted only when consecutive samples have strictly
    opposite signs — this avoids double-counting at exact zero crossings.

    Returns
    -------
    zcr_track : (n_frames,) array in [0, 1]
    """
    signal    = np.asarray(signal, dtype=np.float64)
    frame_len = int(round(frame_sec * sample_rate))
    hop_len   = int(round(hop_sec   * sample_rate))

    frames = _make_frames(signal, frame_len, hop_len)   # (n_frames, frame_len)

    # x[k]*x[k-1] < 0  ⟺  consecutive samples have opposite signs
    crossings = np.sum(
        frames[:, 1:] * frames[:, :-1] < 0, axis=1
    ).astype(np.float64)

    return crossings / (frame_len - 1)


# ---------------------------------------------------------------------------
# Short-term energy
# ---------------------------------------------------------------------------

def frame_energy(
    signal: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
) -> np.ndarray:
    """
    Mean-squared energy per frame.

    E[n] = (1/N) * sum_{k=0}^{N-1} frame[k]^2

    Returns
    -------
    energy : (n_frames,) array, >= 0
    """
    signal    = np.asarray(signal, dtype=np.float64)
    frame_len = int(round(frame_sec * sample_rate))
    hop_len   = int(round(hop_sec   * sample_rate))

    frames = _make_frames(signal, frame_len, hop_len)

    return np.mean(frames ** 2, axis=1)


def log_energy(
    signal: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    floor: float = 1e-10,
) -> np.ndarray:
    """
    Log mean-squared energy per frame.

    log_E[n] = log(max(E[n], floor))

    Returns
    -------
    log_e : (n_frames,) array
    """
    E = frame_energy(signal, sample_rate, frame_sec, hop_sec)
    return np.log(np.maximum(E, floor))


# ---------------------------------------------------------------------------
# Simple voice activity detection
# ---------------------------------------------------------------------------

def speech_activity(
    zcr_track: np.ndarray,
    energy_track: np.ndarray,
    energy_threshold: float | None = None,
    zcr_threshold: float = 0.10,
) -> np.ndarray:
    """
    Classify each frame as SILENCE (0), VOICED (1), or UNVOICED (2).

    Rules
    -----
    silence   :  energy <  energy_threshold
    voiced    :  energy >= energy_threshold  AND  zcr <  zcr_threshold
    unvoiced  :  energy >= energy_threshold  AND  zcr >= zcr_threshold

    Parameters
    ----------
    zcr_track        : (n_frames,) from zcr()
    energy_track     : (n_frames,) from frame_energy()
    energy_threshold : silence cutoff (default: 40 dB below peak energy)
    zcr_threshold    : voiced/unvoiced boundary (default 0.10)

    Returns
    -------
    labels : (n_frames,) int array  —  0=SILENCE, 1=VOICED, 2=UNVOICED
    """
    if energy_threshold is None:
        # 40 dB below peak: threshold = peak * 10^(-40/10) = peak * 1e-4
        peak = np.max(energy_track)
        energy_threshold = peak * 1e-4 if peak > 0 else 0.0

    labels = np.full(len(zcr_track), SILENCE, dtype=np.int8)

    active = energy_track >= energy_threshold
    labels[active & (zcr_track <  zcr_threshold)] = VOICED
    labels[active & (zcr_track >= zcr_threshold)] = UNVOICED

    return labels
