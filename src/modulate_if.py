import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import math

"""
Since MSK is a type of FSK, the amplitude is always between
1 and -1. For convenience, let modulator output be n-bit integer
code representing some value between -1 to 1. Since one code must
represent 0, each LSB represents (1 - (-1))/(2^n - 1) = 2/(2^n - 1)
so for integer code x, the corresponding value in decimal is 
y(x) = -1 + x * 2 / (2^n - 1)

Example: n = 3, so each LSB represents 2/(2^3 - 1) = 2/7
000 -> -1
001 -> -1 + 2/7 = -5/7
010 -> -1 + 4/7 = -3/7
011 -> -1 + 6/7 = -1/7
100 -> -1 + 8/7 = 1/7
101 -> -1 + 10/7 = 3/7
110 -> -1 + 12/7 = 5/7
111 -> -1 + 14/7 = 1

To find x from y, x(y) = round( (y + 1)*(2^n - 1)/2 ), rounding
because x must be an integer

For m samples per symbol, since MSK changes +/- pi/2
per symbol, that means we need to have at least 
1) k-bit wide angle (input to LUT) such that we can differentiate
   between pi/(2m) and pi/2(m+1)
2) Sufficient n such that we can differentiate between
   cos( pi/(2m) ) and cos( pi/(2(m+1)) )

Summary: 
* m samples / symbol so since one symbol increments pi/2,
  each sample increments pi/(2m)
* 1 bit binary input
* (k+j)-bit phase accumulator
* truncate (k+j) bits to k-bit address for LUT
* n-bit LUT output, n-bit DAC input
  (DAC architecture TBD - R-2R or current-steering?)
                                   ┌────────/────────┐                                           
                                   │       k+j       │                                           
 m samples/  ┌────────────┐        │  ┌───────────┐  │                                           
  symbol     │ 0 -> -pi/2 │        ▼  │phase      │  │            ┌───────────┐        ┌────────┐
─────/──────►│ 1 -> pi/2  ├─────/────►│accumulator┼─►┴──────/────►│ cos LUT   ├──/────►│  DAC   │
     1       └────────────┘    k+j    └───────────┘         k     └───────────┘  n     └────────┘

Made with https://asciiflow.com/#/     
"""

@dataclass
class ModulationParams:
    # Bit widths
    n: int = 6
    k: int = 8
    j: int = 0
    m: int = 8

    # Signal
    num_bits: int = 1000
    seed: int = 10086 # for generating bitstream

    # Intermediate Frequency (IF)
    f_s: int = 1000000  # Hz
    f_if: int = 1000000 # Hz

def generate_symbols(num_bits, seed):
    rng = np.random.default_rng(seed)

    # Generate binary bit stream (0 or 1)
    symbols = rng.integers(0, 2, size=num_bits)
    return symbols

def sample_to_phase(sample, params):
    if_increment_rad = (
        2 * math.pi * params.f_if / params.f_s
    )

    # Data-dependent MSK phase advance per sample
    if sample == 1:
        data_increment_rad = math.pi / (2 * params.m)
    elif sample == 0:
        data_increment_rad = -math.pi / (2 * params.m)
    else:
        raise ValueError("Bit stream has values other than 0 or 1")

    # Add both because there is some constant phase incremnt from
    # the IF frequency (not baseband)
    total_increment_rad = if_increment_rad + data_increment_rad

    return rad_to_address(total_increment_rad, params)

def phase_accumulator(symbol_samples, params):
    # Phase accumulator is P bits wide, then truncates phase to k
    # bits before accessing into LUT
    P = params.k + params.j

    # List to store phase accumulated for each symbol sample
    phase_stream = np.zeros(len(symbol_samples), dtype=np.int64)

    # Keep track of accumulating phase
    phase = 0

    for i, samples in enumerate(symbol_samples):
        phase_stream[i] = phase

        phase += sample_to_phase(symbol_samples[i], params)

        # In actual accumulator hardware, sum will wrap around
        # once we've overflowed P = (k + j) bits
        phase %= (1 << P)

    return phase_stream
    
def modulate_msk(symbols, samples_per_symbol, params):
    symbols = generate_symbols(params.num_bits, params.seed)
    symbol_samples = np.repeat(symbols, samples_per_symbol)

    # Need someway to model doing arithmetic in k + j bits
    # then using only k bits to access LUT...
    phase_stream = phase_accumulator(symbol_samples, params)

    # Truncate addresses to k bits (assuming starting from k + j bits)
    lut_addresses = phase_stream >> params.j

    lut = cosine_lut(params)

    output = np.asarray(lut)[lut_addresses]

    return symbols, output

def address_to_rad(address, k):
    # k-bit addresses range from 0 to 2^k - 1 so convert
    # to radians by multiplying by 2*pi / 2^k because
    # angles wrap around at 2*pi, just like address overflows
    # (wraps around) at 2^k = (1 << k)
    prop_constant = 2 * math.pi / (1 << k)

    return prop_constant * address

def rad_to_address(angle_rad, params):
    # Note address_width = (k+j)
    # Addresses range from 0 to 2^(k+j) - 1 so convert
    # radians to address by multiplying by 2^(k+j) / (2*pi) because
    # angles wrap around at 2*pi, just like address overflows
    # (wraps around) at 2^(k+j) = (1 << (k+j))
    prop_constant = (1 << (params.k + params.j))  / (2 * math.pi)

    # address must be integer
    return round(prop_constant * angle_rad)

def cosine_code(angle_rad, n):
    """
    Encode cos(angle) in an n-bit unsigned offset-binary code.
    Code 0 represents -1; code 2^n - 1 represents +1.

    Angle is in radians!!
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    max_code = (1 << n) - 1
    return round((math.cos(angle_rad) + 1) * max_code / 2)
     
def cosine_lut(params):
    """
    Generate a full-cycle cosine LUT with:
      - k-bit address
      - n-bit unsigned offset-binary output

    Returns list of integer codes
    """
    if params.k < 1:
        raise ValueError("k must be >= 1")
    if params.n < 1:
        raise ValueError("n must be >= 1")

    num_entries = 1 << params.k

    return [
        cosine_code(address_to_rad(i, params.k), params.n)
        for i in range(num_entries)
    ]

def decode_lut(codes, n):
    """
    Convert integer codes back to normalized amplitudes in [-1, 1]
    so it's easy to compare between different bit width choices

    Not necessary in actual hardware when feeding integer codes to DAC
    """
    max_code = (1 << n) - 1
    return [2 * code / max_code - 1 for code in codes]

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

            codes = cosine_lut(params)
            amplitudes = decode_lut(codes, n)

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

# Change these lists to sweep different bit widths
plot_quantized_cosines(
    n_values=[6,8,10],
    k_values=[6,8,10],
)