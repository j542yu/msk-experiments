from ModulationParams import ModulationParams
from graphing import plot_quantized_cosines, plot_msk_if_sweep
from modulate_if import generate_symbols, modulate_msk

# Change these lists to sweep different bit widths
# plot_quantized_cosines(
#     n_values=[6,8,10],
#     k_values=[6,8,10],
# )

# Visualize MSK modulated output quantized waveform
# params = ModulationParams()

params = ModulationParams()

symbols = generate_symbols(params.num_bits, params.seed)

symbols, phase_stream, output = modulate_msk(symbols, params)

configurations = [
    {"n": 8, "k": 8, "j": 0},
    {"n": 8, "k": 8, "j": 4},
    {"n": 8, "k": 6, "j": 8},
    {"n": 8, "k": 8, "j": 8},
]

plot_msk_if_sweep(
    base_params=params,
    configurations=configurations,
    num_symbols_to_plot=50,
)