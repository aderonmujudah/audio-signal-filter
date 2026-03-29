function [A, frame_len, hop_len] = lpc_framewise_fp(signal, sample_rate, order, frame_sec, hop_sec, emphasis)
%LPC_FRAMEWISE_FP LPC coefficients per frame via Levinson-Durbin.
% Mirrors `audio_filter/lpc.py` framing + math.
%
% Returns:
%   A (n_frames x order) where rows are [a1..ap]

if nargin < 3 || isempty(order)
    order = 2 + floor(sample_rate / 1000);
end
if nargin < 4 || isempty(frame_sec)
    frame_sec = 0.025;
end
if nargin < 5 || isempty(hop_sec)
    hop_sec = 0.010;
end
if nargin < 6 || isempty(emphasis)
    emphasis = 0.97;
end

x = double(signal(:));

frame_len = round(frame_sec * sample_rate);
hop_len   = round(hop_sec * sample_rate);

if emphasis > 0
    x = local_pre_emphasis(x, emphasis);
end

frames = local_make_frames(x, frame_len, hop_len, true);

n_frames = size(frames, 1);
A = zeros(n_frames, order);

for i = 1:n_frames
    R = local_autocorrelation(frames(i, :).', order);
    A(i, :) = local_levinson_durbin(R).';
end
end

function y = local_pre_emphasis(x, coeff)
if isempty(x)
    y = x;
    return;
end
y = [x(1); x(2:end) - coeff .* x(1:end-1)];
end

function frames = local_make_frames(x, frame_len, hop_len, apply_window)
center_pad = floor(frame_len / 2);
x = [zeros(center_pad, 1); x; zeros(center_pad, 1)];

n_frames  = 1 + floor((length(x) - frame_len + hop_len - 1) / hop_len);
pad_right = (n_frames - 1) * hop_len + frame_len - length(x);
if pad_right > 0
    x = [x; zeros(pad_right, 1)];
end

idx = (0:frame_len-1)' + (0:n_frames-1) * hop_len;
frames = x(idx + 1).';

if apply_window
    w = hann_window_periodic(frame_len);
    frames = frames .* w.';
end
end

function R = local_autocorrelation(frame, order)
N = length(frame);
R = zeros(order + 1, 1);
for k = 0:order
    R(k + 1) = (frame(1:N-k)' * frame(1+k:N)) / N;
end
end

function a = local_levinson_durbin(R)
order = length(R) - 1;

if R(1) < 1e-10
    a = zeros(order, 1);
    return;
end

a = zeros(order, 1);
E = double(R(1));

for i = 1:order
    if i == 1
        dot_term = 0.0;
    else
        dot_term = a(1:i-1).' * R(i:-1:2);
    end

    k = -(R(i + 1) + dot_term) / E;

    if i > 1
        a_prev = a(1:i-1);
        a(1:i-1) = a_prev + k .* flipud(a_prev);
    end

    a(i) = k;
    E = E * (1.0 - k * k);

    if E <= 1e-10
        break;
    end
end
end
