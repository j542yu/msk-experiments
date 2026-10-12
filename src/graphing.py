import matplotlib.pyplot as plt
import math
import numpy as np

import lut_logic
from ModulationParams import ModulationParams
from modulate_if import modulate_msk

# Visualize effect of bit width choices on cos wave in LUT
def plot_quantized_cosines(n_values, k_values):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Ideal cosine reference
    phase_ideal = [
        2 * math.pi * i / 2000 for i in range(2001)
    ]
    ax.plot(
        phase_ideal,
        [math.cos(p) for p in phase_ideal],
        "k--",
        label="Ideal cosine",
    )

    # Plot each combination of n and k
    for n in n_values:
        for k in k_values:
            params = ModulationParams(n=n, k=k, j=0)

            codes = lut_logic.cosine_lut(params)
            amplitudes = lut_logic.decode_lut(codes, n)

            phases = [
                2 * math.pi * i / (1 << k)
                for i in range(1 << k)
            ]

            # Extend the last step to 2*pi
            phases.append(2 * math.pi)
            amplitudes.append(amplitudes[0])

            ax.step(
                phases,
                amplitudes,
                where="post",
                label=f"n={n}, k={k}",
            )

    ax.set_xlabel("Phase (radians)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Quantized Cosine LUT Output")
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks(
        [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi],
        ["0", "π/2", "π", "3π/2", "2π"],
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    make_legend_interactive(fig, ax)
    fig.tight_layout()
    plt.show()

# Visualize modulation result
def plot_msk_if(params, num_symbols_to_plot=12):
    symbols, phase_stream, output = modulate_msk(params)

    # Normalize unsigned DAC codes to [-1, 1] for easier
    # comparison when varying params but not needed in actual HW
    waveform = np.asarray(lut_logic.decode_lut(output, params.n))

    # Time associated with each output sample
    time = np.arange(len(waveform)) / params.f_s

    # Only plot the first few symbols so the oscillations are visible.
    num_samples = min(
        num_symbols_to_plot * params.m,
        len(waveform)
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True
    )

    # Plot the quantized IF waveform.
    ax1.step(time[:num_samples] * 1e6, waveform[:num_samples])
    ax1.set_ylabel("Normalized amplitude")
    ax1.set_title("MSK IF Output")
    ax1.grid(True, alpha=0.3)

    # Show the input bits as a step waveform.
    bit_times = np.arange(
        num_symbols_to_plot + 1
    ) * params.m / params.f_s * 1e6

    shown_bits = symbols[:num_symbols_to_plot]

    ax2.step(
        bit_times[:len(shown_bits)],
        shown_bits,
        where="post",
    )
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Input bit")
    ax2.set_yticks([0, 1])
    ax2.set_title("Input Data")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

    return symbols, waveform, phase_stream

def make_legend_interactive(fig, ax):
    legend = ax.get_legend()
    if legend is None:
        return

    # Map legend entries to their plotted lines
    legend_lines = legend.get_lines()
    plotted_lines = ax.get_lines()

    # First plotted line is the ideal cosine
    for legend_line, plotted_line in zip(legend_lines, plotted_lines):
        legend_line.set_picker(True)
        legend_line.set_pickradius(5)

    def on_pick(event):
        legend_line = event.artist
        index = legend_lines.index(legend_line)
        plotted_line = plotted_lines[index]

        visible = not plotted_line.get_visible()
        plotted_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
