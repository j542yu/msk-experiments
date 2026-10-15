import matplotlib.pyplot as plt
import math
import numpy as np

import lut_logic
from ModulationParams import ModulationParams
from modulate_if import modulate_msk, generate_symbols
from measure_error import ideal_msk_waveform_oversampled, dac_zero_order_hold

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

def plot_msk_if_sweep(
    base_params,
    configurations,
    num_symbols_to_plot=12,
):
    """
    Overlay the ideal MSK waveform and modulator outputs for
    multiple n, k, j configurations.

    The same input symbols and timing parameters are used for
    every configuration.

    Args:
        base_params: Base ModulationParams instance.
        configurations: List of dictionaries, each containing
            n, k, and j.
        num_symbols_to_plot: Number of symbols to display.
    """
    # Generate one common input sequence for every configuration.
    symbols = generate_symbols(
        base_params.num_bits,
        base_params.seed,
    )

    # Create a shared ideal MSK reference waveform.
    ideal = ideal_msk_waveform_oversampled(
        symbols,
        base_params,
        oversample=16,
    )

    # Time axis for the ideal waveform, in microseconds.
    ideal_time = (
        np.arange(len(ideal))
        / (16 * base_params.f_s)
        * 1e6
    )

    # Time axis for the actual modulator outputs.
    sample_time = (
        np.arange(len(symbols) * base_params.m)
        / base_params.f_s
        * 1e6
    )

    # Limit all plots to the same number of symbols.
    num_samples = min(
        len(sample_time),
        num_symbols_to_plot * base_params.m,
    )

    num_ideal_points = num_samples * 16

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Plot the ideal MSK waveform.
    ax1.plot(
        ideal_time[:num_ideal_points],
        ideal[:num_ideal_points],
        "k--",
        linewidth=2,
        label="Ideal MSK",
    )

    # Overlay the modulator output for each configuration.
    for config in configurations:
        # Keep timing and input settings fixed.
        params = ModulationParams(
            n=config["n"],
            k=config["k"],
            j=config["j"],
            m=base_params.m,
            num_bits=base_params.num_bits,
            seed=base_params.seed,
            f_s=base_params.f_s,
            f_if=base_params.f_if,
        )

        # Generate the actual LUT-based modulator output.
        _, _, output = modulate_msk(symbols, params)

        # Convert DAC codes to normalized amplitudes [-1, +1].
        waveform = np.asarray(
            lut_logic.decode_lut(output, params.n)
        )

        ax1.step(
            sample_time[:num_samples],
            waveform[:num_samples],
            where="post",
            linewidth=1.3,
            alpha=0.85,
            label=f"n={params.n}, k={params.k}, j={params.j}",
        )

    # Plot the input symbols in the lower panel.
    shown_symbols = symbols[:num_symbols_to_plot]
    bit_times = (
        np.arange(len(shown_symbols) + 1)
        * base_params.m
        / base_params.f_s
        * 1e6
    )

    ax2.step(
        bit_times[:-1],
        shown_symbols,
        where="post",
        color="tab:blue",
        linewidth=1.8,
        label="Input symbols",
    )

    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_ylabel("Bit")
    ax2.set_xlabel("Time (µs)")
    ax2.set_title("Transmitted Data")
    ax2.grid(True, alpha=0.3)

    # Configure the overlaid waveform plot.
    ax1.set_ylabel("Normalized amplitude")
    ax1.set_title("Ideal MSK vs. Modulator Outputs")
    ax1.grid(True, alpha=0.3)

    # Allow waveform configurations to be toggled via the legend.
    make_legend_interactive(fig, ax1)

    fig.tight_layout()
    plt.show()

def make_legend_interactive(fig, ax):
    legend = ax.get_legend()
    if legend is None:
        return

    # Map legend entries to their plotted lines
    legend_lines = legend.get_lines()
    plotted_lines = ax.get_lines()

    # First plotted line is the ideal cosine
    for legend_line, _ in zip(legend_lines, plotted_lines):
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

def plot_dac_vs_ideal(
    symbols,
    params,
    dac_codes,
    oversample=16,
    num_symbols_to_plot=12,
):
    """
    Plot the ideal MSK waveform against the zero-order-hold
    DAC output.

    Args:
        symbols: Original binary symbol sequence.
        params: ModulationParams instance.
        dac_codes: DAC output decoded to normalized amplitude,
                   typically between -1 and +1.
        oversample: Plotting time resolution per DAC sample.
        num_symbols_to_plot: Number of initial symbols to display.
    """
    # Generate both waveforms on the same fine time grid.
    ideal = ideal_msk_waveform_oversampled(
        symbols, params, oversample
    )
    dac_output = dac_zero_order_hold(
        dac_codes, oversample
    )

    if ideal.shape != dac_output.shape:
        raise ValueError(
            "Waveforms have different lengths. Check that "
            "dac_codes contains one code per original sample."
        )

    # Convert the fine-grid sample indices into seconds.
    # Each original sample period is divided into oversample steps.
    time = np.arange(len(ideal)) / (
        oversample * params.f_s
    )

    # Limit the plot to the requested number of symbols.
    num_points = min(
        len(ideal),
        num_symbols_to_plot * params.m * oversample,
    )

    time_plot = time[:num_points]
    ideal_plot = ideal[:num_points]
    dac_plot = dac_output[:num_points]

    # Plot both waveforms.
    plt.figure(figsize=(12, 5))

    plt.plot(
        time_plot * 1e6,
        ideal_plot,
        label="Ideal MSK waveform",
        linewidth=2,
    )

    plt.step(
        time_plot * 1e6,
        dac_plot,
        where="post",
        label="DAC output (zero-order hold)",
        linewidth=1.5,
    )

    plt.xlabel("Time (µs)")
    plt.ylabel("Normalized amplitude")
    plt.title("Ideal MSK waveform vs. DAC output")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_msk_phase(phase_stream, params, num_symbols=10):
    P = params.k + params.j
    phase_scale = 2 * np.pi / (1 << P)

    # Convert phase counts into radians.
    phase_rad = np.asarray(phase_stream) * phase_scale

    # Unwrap the phase so accumulator rollover does not look
    # like a sudden negative jump of approximately 2*pi.
    phase_unwrapped = np.unwrap(phase_rad)

    # Limit the plot to a short section.
    num_samples = min(
        len(phase_unwrapped),
        num_symbols * params.m,
    )

    sample_index = np.arange(num_samples)

    plt.figure(figsize=(12, 5))
    plt.plot(
        sample_index,
        phase_unwrapped[:num_samples],
        marker=".",
        markersize=3,
    )
    plt.xlabel("Sample index")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title("MSK Accumulator Phase")
    plt.grid(True)
    plt.tight_layout()
    plt.show()