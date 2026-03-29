% Minimal MATLAB smoke tests for the reference implementations.

addpath(fileparts(mfilename('fullpath')));

sr = 16000;
dur = 1.0;
freq = 440.0;
t = (0:round(dur*sr)-1)' / sr;
x = sin(2*pi*freq*t);

% STFT -> ISTFT perfect reconstruction
[S, frame_len, hop_len] = stft_fp(x, sr);
y = istft_fp(S, sr, frame_len, hop_len, length(x));

max_err = max(abs(x - y));
fprintf('[PASS] STFT/ISTFT perfect reconstruction  max_err=%.2e\n', max_err);
assert(max_err < 1e-10);

% MFCC shape
c = mfcc_fp(x, sr);
assert(size(c, 2) == 13);
fprintf('[PASS] MFCC shape  %d frames x %d coeffs\n', size(c,1), size(c,2));

% LPC -> formants shape
[A, ~, ~] = lpc_framewise_fp(x, sr);
[f1, bw1] = lpc_to_formants_fp(A, sr);
assert(all(size(f1) == size(bw1)));
fprintf('[PASS] LPC->formants shape  %d frames x %d formants\n', size(f1,1), size(f1,2));

fprintf('\nAll MATLAB smoke tests passed.\n');
