# Audio Filter — Project Context

## What this project is

A small DSP-focused Python library (NumPy-first, from first principles) with matching MATLAB reference implementations.

Current scope:

- STFT / ISTFT (Hann window, overlap-add, normalization)
- MFCCs (mel filterbank + orthonormal DCT-II) + delta features
- LPC (autocorrelation + Levinson–Durbin) + formant extraction via polynomial roots

## MATLAB role

MATLAB (`matlab/`) is the prototyping + validation layer.

When results look off in Python, the MATLAB version is the source of truth to diff against.
To avoid name collisions with MATLAB built-ins (e.g. `stft`), the MATLAB files use an `_fp` suffix.

## Math constraints

Everything numerical should be from first principles:

- No sklearn/librosa/torch or pretrained models
- NumPy FFTs are allowed
- SciPy is allowed for basic utilities / I/O (as needed)

## Repo structure

```
Audio Filter/
├── audio_filter/          # Python implementations
│   ├── stft.py
│   ├── mfcc.py
│   └── lpc.py
├── matlab/                # MATLAB reference implementations
│   ├── stft_fp.m
│   ├── istft_fp.m
│   ├── mfcc_fp.m
│   ├── lpc_framewise_fp.m
│   ├── lpc_to_formants_fp.m
│   └── run_smoke_tests.m
├── tests/                 # Python tests
└── README.md
```
