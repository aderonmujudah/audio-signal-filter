function [freqs, bws] = lpc_to_formants_fp(A, sample_rate, n_formants, min_freq, max_bw)
%LPC_TO_FORMANTS_FP Formant extraction from per-frame LPC coefficients.
% Mirrors `audio_filter/lpc.py:lpc_to_formants`.

if nargin < 3 || isempty(n_formants)
    n_formants = 4;
end
if nargin < 4 || isempty(min_freq)
    min_freq = 90.0;
end
if nargin < 5 || isempty(max_bw)
    max_bw = 400.0;
end

n_frames = size(A, 1);
freqs = nan(n_frames, n_formants);
bws   = nan(n_frames, n_formants);

for i = 1:n_frames
    poly = [1.0, A(i, :)];
    r = roots(poly);

    r = r(imag(r) >= 0);
    if isempty(r)
        continue;
    end

    angles = angle(r);
    f = angles * sample_rate / (2.0 * pi);
    bw = -log(abs(r)) * sample_rate / pi;

    mask = (f > min_freq) & (bw > 0) & (bw < max_bw);
    f = f(mask);
    bw = bw(mask);

    if isempty(f)
        continue;
    end

    [f_sorted, idx] = sort(f);
    bw_sorted = bw(idx);

    n = min(length(f_sorted), n_formants);
    freqs(i, 1:n) = f_sorted(1:n);
    bws(i, 1:n)   = bw_sorted(1:n);
end
end
