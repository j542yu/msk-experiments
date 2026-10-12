import matplotlib.pyplot as plt
from ModulationParams import ModulationParams
from graphing import plot_quantized_cosines, plot_msk_if

# Change these lists to sweep different bit widths
plot_quantized_cosines(
    n_values=[6,8,10],
    k_values=[6,8,10],
)

# Visualize MSK modulated output quantized waveform
params = ModulationParams()

symbols, waveform, phase_stream = plot_msk_if(params)