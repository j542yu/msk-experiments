import matplotlib.pyplot as plt
import numpy as np

from ModulationParams import ModulationParams
from graphing import plot_quantized_cosines, plot_msk_if
from modulate_if import generate_symbols, modulate_msk, phase_accumulator
from measure_error import measure_dac_snr
from lut_logic import decode_lut

import math
from dataclasses import replace

# Change these lists to sweep different bit widths
# plot_quantized_cosines(
#     n_values=[6,8,10],
#     k_values=[6,8,10],
# )

# Visualize MSK modulated output quantized waveform
# params = ModulationParams()
# plot_msk_if(params)

# Check correlation with ideal signal to measure error
base_params = ModulationParams(
    n=8,
    k=16,
    j=8,
    m=8,
    num_bits=40000, #5kB
    seed=10086,
    f_s=1_000_000,
    f_if=100_000,
)

symbols = generate_symbols(base_params.num_bits, base_params.seed)

# Run your existing MSK modulator.
_, phase_stream, output_codes = modulate_msk(symbols, base_params)

# Convert unsigned LUT codes into normalized amplitudes in [-1, +1].
dac_samples = np.asarray(
    decode_lut(output_codes, base_params.n),
    dtype=float
)

# Compare the reconstructed DAC output against ideal continuous-time MSK.
metrics = measure_dac_snr(
    symbols,
    base_params,
    dac_samples,
    oversample=32
)

print(f"DAC waveform RMSE: {metrics['rmse']:.8f}")
print(f"DAC waveform max error: {metrics['max_error']:.8f}")
print(f"DAC waveform SNR: {metrics['snr_db']:.3f} dB")