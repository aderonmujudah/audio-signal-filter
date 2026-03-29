function f = mel_to_hz_fp(m)
%MEL_TO_HZ_FP Convert mel to Hz (matches Python).

f = 700.0 .* (10.0 .^ (double(m) ./ 2595.0) - 1.0);
end
