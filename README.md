# Audio Filter (Python + MATLAB reference)

DSP building blocks implemented from first principles in **Python/NumPy**, with matching **MATLAB** reference implementations kept in-repo for validation.

## What’s in here

- Python package: `audio_filter/`
  - `stft.py` — STFT/ISTFT with Hann window + overlap-add normalization
  - `mfcc.py` — MFCCs (mel filterbank + DCT-II) + delta features
  - `lpc.py` — LPC via Levinson–Durbin + formant extraction
- MATLAB reference: `matlab/`
  - `stft_fp.m`, `istft_fp.m`
  - `mfcc_fp.m` (+ mel/DCT helpers)
  - `lpc_framewise_fp.m`, `lpc_to_formants_fp.m`

## Quick start (Python)

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Quick start (MATLAB)

From the repo root in MATLAB:

```matlab
addpath('matlab');
run('matlab/run_smoke_tests.m');
```

## Dev principle

MATLAB is the “source of truth” for algorithm validation. If outputs diverge, diff against the MATLAB implementations first, then adjust Python.
