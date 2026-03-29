function D = dct_matrix_fp(n_input, n_output)
%DCT_MATRIX_FP Orthonormal DCT-II matrix (matches `audio_filter/mfcc.py`).
% Shape: (n_output x n_input)

m = 0:(n_input - 1);
k = (0:(n_output - 1))';

D = cos(pi .* k .* (m + 0.5) ./ n_input);

D(1, :) = D(1, :) .* (1.0 / sqrt(n_input));
if n_output > 1
    D(2:end, :) = D(2:end, :) .* sqrt(2.0 / n_input);
end
end
