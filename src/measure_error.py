import numpy as np


def calculate_snr(reference, measured):
    """
    Calculate signal-to-error ratio between two waveforms.

    Both arrays must represent the same signal over the same
    time interval, sampled at the same instants.

    Returns:
        Dictionary containing signal power, error power,
        RMSE, maximum absolute error, and SNR in dB.
    """
    reference = np.asarray(reference, dtype=float)
    measured = np.asarray(measured, dtype=float)

    if reference.shape != measured.shape:
        raise ValueError("Waveforms must have the same shape.")

    if reference.size == 0:
        raise ValueError("Waveforms cannot be empty.")

    # Difference between the measured and ideal waveforms.
    error = measured - reference

    # Average squared amplitudes represent average signal power
    # when both waveforms use the same impedance and scaling.
    signal_power = np.mean(reference**2)
    error_power = np.mean(error**2)

    rmse = np.sqrt(error_power)
    max_error = np.max(np.abs(error))

    # Handle zero-power cases explicitly.
    if error_power == 0:
        snr_db = float("inf")
    elif signal_power == 0:
        snr_db = float("-inf")
    else:
        snr_db = 10 * np.log10(signal_power / error_power)

    return {
        "signal_power": signal_power,
        "error_power": error_power,
        "rmse": rmse,
        "max_error": max_error,
        "snr_db": snr_db,
    }


def ideal_msk_waveform_oversampled(symbols, params, oversample=16):
    """
    Generate an ideal MSK waveform at a higher time resolution.

    Each original sample interval is divided into 'oversample'
    subintervals. The phase advances continuously through each
    original sample interval.

    The returned waveform contains:
        len(symbols) * params.m * oversample samples.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")

    # Repeat each data symbol for the configured samples per symbol.
    symbol_samples = np.repeat(np.asarray(symbols), params.m)

    # Calculate the ideal phase increment during each original
    # sample interval, including the IF carrier contribution.
    phase_increment = (
        2 * np.pi * params.f_if / params.f_s
        + (2 * symbol_samples - 1) * np.pi / (2 * params.m)
    )

    # Each original interval is divided into smaller time steps.
    # The phase at the start of the first interval is zero.
    fractional_increment = phase_increment / oversample

    # Build the phase at each sub-sample instant.
    # The first sub-sample has phase zero, matching the original
    # modulator's pre-increment phase convention.
    phase_steps = np.repeat(fractional_increment, oversample)

    phase = np.zeros(len(phase_steps), dtype=float)

    if len(phase) > 1:
        phase[1:] = np.cumsum(phase_steps[:-1])

    return np.cos(phase)


def dac_zero_order_hold(samples, oversample=16):
    """
    Model a zero-order-hold DAC.

    Each DAC code is held constant for one original sample period.
    Repeating each sample creates a higher-resolution representation
    of the staircase waveform.

    This models the DAC hold behavior, but not an analog
    reconstruction filter.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")

    samples = np.asarray(samples, dtype=float)

    return np.repeat(samples, oversample)


def measure_dac_snr(symbols, params, dac_codes, oversample=16):
    """
    Compare the ideal continuous-time MSK waveform against the
    waveform produced by a zero-order-hold DAC.

    dac_codes should be the modulator's output codes decoded into
    normalized amplitude units, such as values from -1 to +1.
    """
    # Approximate the continuous-time ideal waveform.
    ideal = ideal_msk_waveform_oversampled(
        symbols, params, oversample
    )

    # Approximate the DAC's held output at the same time instants.
    dac_output = dac_zero_order_hold(dac_codes, oversample)

    if ideal.shape != dac_output.shape:
        raise ValueError(
            "Ideal waveform and DAC output have different lengths. "
            "Check symbols, DAC codes, and samples per symbol."
        )

    return calculate_snr(ideal, dac_output)