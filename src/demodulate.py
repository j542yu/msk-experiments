import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from modulate import gaussian_filter


def apply_gaussian_filter(signal, BT, samples_per_symbol, span_symbols=4):

    h = gaussian_filter(
        BT,
        samples_per_symbol,
        span_symbols
    )

    return fftconvolve(signal, h, mode="same")

def differential_detector(signal, samples_per_symbol):

    N = samples_per_symbol

    current = signal[N:]
    delayed = signal[:-N]

    differential = current * np.conj(delayed)

    return differential

def sample_symbols(
    differential,
    samples_per_symbol,
    sampling_offset=None
):
    """
    Extract one differential-detector output per symbol.
    """

    N = samples_per_symbol

    if sampling_offset is None:
        # Start in the middle of each symbol
        sampling_offset = N // 2

    samples = differential[
        sampling_offset::N
    ]

    return samples

def symbol_decision(symbol_samples):

    decision_metric = np.imag(symbol_samples)

    detected_bits = (decision_metric > 0).astype(int)

    return detected_bits, decision_metric

def gmsk_msk_demodulator(
    signal,
    samples_per_symbol,
    filter_BT=0.5,
    filter_span=4,
    sampling_offset=None
):

    # --------------------------------
    # 1. Gaussian receiver filter
    # --------------------------------

    filtered = signal

    # filtered = apply_gaussian_filter(
    #     signal,
    #     filter_BT,
    #     samples_per_symbol,
    #     filter_span
    # )

    # --------------------------------
    # 2. One-symbol differential
    # --------------------------------

    differential = differential_detector(
        filtered,
        samples_per_symbol
    )

    # --------------------------------
    # 3. Symbol sampling
    # --------------------------------

    symbol_samples = sample_symbols(
        differential,
        samples_per_symbol,
        sampling_offset
    )

    # --------------------------------
    # 4. Symbol decision
    # --------------------------------

    detected_bits, metric = symbol_decision(
        symbol_samples
    )

    return {
        "filtered": filtered,
        "differential": differential,
        "symbol_samples": symbol_samples,
        "decision_metric": metric,
        "detected_bits": detected_bits,
    }