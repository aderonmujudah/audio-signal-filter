function [f0_track, conf_track] = pitch_fp(signal, sample_rate, f0_min, f0_max, frame_sec, hop_sec, voiced_threshold)
%PITCH_FP Per-frame F0 estimation via normalized autocorrelation.
% Mirrors `audio_filter/pitch.py`.
%
% Returns:
%   f0_track   (n_frames x 1)  Hz, NaN for unvoiced frames
%   conf_track (n_frames x 1)  autocorr peak height in [0,1]

if nargin < 3 || isempty(f0_min),            f0_min  = 60.0;  end
if nargin < 4 || isempty(f0_max),            f0_max  = 400.0; end
if nargin < 5 || isempty(frame_sec),         frame_sec = 0.025; end
if nargin < 6 || isempty(hop_sec),           hop_sec   = 0.010; end
if nargin < 7 || isempty(voiced_threshold),  voiced_threshold = 0.45; end

x         = double(signal(:));
frame_len = round(frame_sec * sample_rate);
hop_len   = round(hop_sec   * sample_rate);

tau_min = max(1,            floor(sample_rate / f0_max));
tau_max = min(frame_len-1,  ceil( sample_rate / f0_min));

frames   = local_make_frames(x, frame_len, hop_len);
n_frames = size(frames, 1);

f0_track   = NaN(n_frames, 1);
conf_track = zeros(n_frames, 1);

for i = 1:n_frames
    [f0, conf] = local_pitch_frame(frames(i,:).', sample_rate, tau_min, tau_max, voiced_threshold);
    f0_track(i)   = f0;
    conf_track(i) = conf;
end
end


% ── framing ──────────────────────────────────────────────────────────────────
function frames = local_make_frames(x, frame_len, hop_len)
center_pad = floor(frame_len / 2);
x = [zeros(center_pad,1); x; zeros(center_pad,1)];

n_frames  = 1 + floor((length(x) - frame_len + hop_len - 1) / hop_len);
pad_right = (n_frames - 1) * hop_len + frame_len - length(x);
if pad_right > 0
    x = [x; zeros(pad_right, 1)];
end

idx    = (0:frame_len-1)' + (0:n_frames-1) * hop_len;
frames = x(idx + 1).';                  % (n_frames x frame_len) — no Hann window
end


% ── normalized autocorrelation ───────────────────────────────────────────────
function r = local_autocorr(frame, max_lag)
R0 = frame' * frame;
r  = zeros(max_lag + 1, 1);
if R0 < 1e-10
    return;
end
r(1) = 1.0;
for k = 1:max_lag
    r(k+1) = (frame(1:end-k)' * frame(1+k:end)) / R0;
end
end


% ── parabolic interpolation ──────────────────────────────────────────────────
function tau = local_parabolic(r, peak_idx)
% peak_idx is 1-based
tau = double(peak_idx);
if peak_idx <= 1 || peak_idx >= length(r)
    return;
end
y0 = r(peak_idx-1);  y1 = r(peak_idx);  y2 = r(peak_idx+1);
denom = y0 - 2*y1 + y2;
if abs(denom) < 1e-12
    return;
end
% convert to 0-based offset then back: offset in 0-based = peak_idx-1 + shift
tau = (peak_idx - 1) - 0.5*(y2 - y0)/denom;   % 0-based fractional lag
end


% ── single-frame estimator ───────────────────────────────────────────────────
function [f0, conf] = local_pitch_frame(frame, sample_rate, tau_min, tau_max, thr)
r = local_autocorr(frame, tau_max);

% Search within [tau_min, tau_max]  (1-based: tau_min+1 .. tau_max+1)
segment   = r(tau_min+1 : tau_max+1);
[conf, local_peak] = max(segment);
global_1  = local_peak + tau_min;   % 1-based index into r

if conf < thr
    f0 = NaN;
    return;
end

tau_exact = local_parabolic(r, global_1);   % 0-based fractional lag

if tau_exact <= 0
    f0 = NaN;
    return;
end

f0 = sample_rate / tau_exact;
end
