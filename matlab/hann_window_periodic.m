function w = hann_window_periodic(n)
%HANN_WINDOW_PERIODIC Periodic Hann window (matches Python implementation).
%   w[k] = 0.5 * (1 - cos(2*pi*k/n)),  k = 0..n-1

k = (0:n-1)';
w = 0.5 * (1 - cos(2*pi*k/n));
end
