import numpy as np
from scipy.signal import fftconvolve
from dataclasses import dataclass # for convenience when defining class

"""
Purpose: Experiment with various parameters when modulating
a bit stream to MSK or GMSK so we can then feed this to a Python
model of what we plan to do in RTL to simulate the accuracy of
demodulating before deciding on specific parameters for the RTL
"""

@dataclass
class SimulationParams:
    # ------------- Signal ------------- #
    num_bits: int = 1000
    samples_per_symbol: int = 8

    # MSK or GMSK   
    modulation: str = "MSK"     

    # Bandwidth-symbol time product
    # Only applicable to GMSK; for more info see
    # https://dsp.stackexchange.com/questions/30653/what-is-a-bt-bandwidth-time-product-with-reference-to-modulation
    BT: float = 0.3

    # Gaussian filter length in symbols
    gaussian_span: int = 4

    # ----------- Impairments ----------- #
    EbN0_dB: float = np.inf

    # As fraction of symbol rate
    freq_offset: float = 0.0

    # As fraction of symbol period
    timing_offset: float = 0.0

    # Arbitrary initial carrier phase (rad)
    phase_offset: float = 0.0

    # ----------- Quantization ----------- #
    # None = floating point
    # integer N = signed N-bit I/Q
    quant_bits: int | None = None

    # ----------- Randomness ----------- #
    seed: int = 12345

def generate_bit_stream(num_bits, rng):
    # Generate binary bit stream (0 or 1)
    bits = rng.integers(0, 2, size=num_bits)

    # Convert 0 to -1, 1 to 1 for MSK
    symbols = 2*bits - 1

    return bits, symbols

def gaussian_filter(BT, samples_per_symbol, span_symbols):
    N = samples_per_symbol

    t = np.arange(
        -span_symbols*N/2,
        span_symbols*N/2 + 1
    ) / N

    B = BT/1

    h = np.sqrt(2*np.pi)*B*np.exp(
        -2 * np.pi**2 * B**2 * t**2
    )

    h /= np.sum(h)

    return h

def add_awgn(signal, EbN0_dB, samples_per_symbol, rng):

    if np.isinf(EbN0_dB):
        return signal.copy()

    N = samples_per_symbol

    EbN0 = 10 ** (EbN0_dB / 10)

    # For unit-energy symbols and N samples/symbol:
    #
    # variance of each real noise component:
    #
    # N / (2 * Eb/N0)

    noise_std = np.sqrt(
        N / (2 * EbN0)
    )

    noise = noise_std * (
        rng.standard_normal(len(signal))
        + 1j * rng.standard_normal(len(signal))
    )

    return signal + noise

def generate_msk_gmsk(symbols, params):
    """
    Generate complex baseband MSK/GMSK waveform.

    Returns:
        complex numpy array containing I + jQ
    """

    N = params.samples_per_symbol

    # Upsample symbol sequence
    symbol_samples = np.repeat(symbols, N)

    # ----------------------------------
    # Frequency shaping
    # ----------------------------------

    if params.modulation.upper() == "MSK":

        # Rectangular frequency pulse
        shaped = symbol_samples.astype(float)

    elif params.modulation.upper() == "GMSK":

        gaussian = gaussian_filter(
            params.BT,
            N,
            params.gaussian_span
        )

        shaped = fftconvolve(
            symbol_samples,
            gaussian,
            mode="same"
        )

    else:
        raise ValueError(
            "modulation must be 'MSK' or 'GMSK'"
        )

    # ----------------------------------
    # Integrate instantaneous frequency
    # ----------------------------------

    #
    # MSK modulation index:
    #
    # h = 1/2
    #
    # Phase increment:
    #
    # dphi = pi*h*a[n]/N
    #

    h = 0.5

    phase_increment = np.pi * h * shaped / N

    phase = np.cumsum(phase_increment)

    # Initial phase
    phase += params.phase_offset

    # Complex baseband signal
    signal = np.exp(1j * phase)

    return signal

def add_frequency_offset(signal, frequency_offset, samples_per_symbol):

    N = samples_per_symbol

    n = np.arange(len(signal))

    # Sampling rate in units of symbol rate
    Fs = N

    # Frequency offset is specified as fraction of symbol rate
    delta_f = frequency_offset

    rotation = np.exp(
        1j * 2 * np.pi * delta_f * n / Fs
    )

    return signal * rotation

def add_timing_offset(signal, timing_offset, samples_per_symbol):

    if timing_offset == 0:
        return signal.copy()

    N = samples_per_symbol

    # Timing offset measured as fraction of a symbol
    shift_samples = timing_offset * N

    old_indices = np.arange(len(signal))
    new_indices = old_indices + shift_samples

    I = np.interp(
        new_indices,
        old_indices,
        np.real(signal),
        left=0.0,
        right=0.0
    )

    Q = np.interp(
        new_indices,
        old_indices,
        np.imag(signal),
        left=0.0,
        right=0.0
    )

    return I + 1j * Q

def quantize_iq(signal, bits):

    if bits is None:
        return signal.copy()

    if bits < 1:
        raise ValueError("quantization bits must be >= 1")

    max_int = 2**(bits - 1) - 1
    min_int = -2**(bits - 1)

    # Since |signal| = 1 ideally, I and Q are already
    # approximately in [-1, +1].
    I = np.real(signal)
    Q = np.imag(signal)

    I_q = np.round(I * max_int)
    Q_q = np.round(Q * max_int)

    # Saturate
    I_q = np.clip(I_q, min_int, max_int)
    Q_q = np.clip(Q_q, min_int, max_int)

    return I_q.astype(np.int32), Q_q.astype(np.int32)

def generate_test_signal(params):

    rng = np.random.default_rng(params.seed)

    # -----------------------------
    # Generate bits
    # -----------------------------

    bits, symbols = generate_bit_stream(
        params.num_bits,
        rng
    )

    # -----------------------------
    # Modulate
    # -----------------------------

    signal = generate_msk_gmsk(
        symbols,
        params
    )

    # -----------------------------
    # Frequency offset
    # -----------------------------

    signal = add_frequency_offset(
        signal,
        params.freq_offset,
        params.samples_per_symbol
    )

    # -----------------------------
    # Timing offset
    # -----------------------------

    signal = add_timing_offset(
        signal,
        params.timing_offset,
        params.samples_per_symbol
    )

    # -----------------------------
    # AWGN
    # -----------------------------

    signal = add_awgn(
        signal,
        params.EbN0_dB,
        params.samples_per_symbol,
        rng
    )

    # -----------------------------
    # Quantization
    # -----------------------------

    if params.quant_bits is None:

        I = np.real(signal)
        Q = np.imag(signal)

    else:

        I, Q = quantize_iq(
            signal,
            params.quant_bits
        )

    return {
        "bits": bits,
        "symbols": symbols,
        "signal": signal,
        "I": I,
        "Q": Q,
    }
