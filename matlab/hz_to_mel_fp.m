function m = hz_to_mel_fp(f)
%HZ_TO_MEL_FP Convert Hz to mel (matches Python).

m = 2595.0 .* log10(1.0 + double(f) ./ 700.0);
end
