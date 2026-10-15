import matplotlib.pyplot as plt
from ModulationParams import ModulationParams
from graphing import plot_quantized_cosines, plot_msk_if
from lut_logic import cosine_lut


# Change these lists to sweep different bit widths
# plot_quantized_cosines(
#     n_values=[8],
#     k_values=[5,8,10],
# )

# # Visualize MSK modulated output quantized waveform
params = ModulationParams()

# print(cosine_lut(params))

symbols, waveform, phase_stream = plot_msk_if(params)