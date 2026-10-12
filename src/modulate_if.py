import numpy as np
import math

import lut_logic

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
    
def modulate_msk(params):
    symbols = generate_symbols(params.num_bits, params.seed)
    symbol_samples = np.repeat(symbols, params.m)

    # Need someway to model doing arithmetic in k + j bits
    # then using only k bits to access LUT...
    phase_stream = phase_accumulator(symbol_samples, params)

    # Truncate addresses to k bits (assuming starting from k + j bits)
    lut_addresses = phase_stream >> params.j

    lut = lut_logic.cosine_lut(params)

    output = np.asarray(lut)[lut_addresses]

    return symbols, phase_stream, output

def rad_to_address(angle_rad, params):
    # Note address_width = (k+j)
    # Addresses range from 0 to 2^(k+j) - 1 so convert
    # radians to address by multiplying by 2^(k+j) / (2*pi) because
    # angles wrap around at 2*pi, just like address overflows
    # (wraps around) at 2^(k+j) = (1 << (k+j))
    prop_constant = (1 << (params.k + params.j))  / (2 * math.pi)

    # address must be integer
    return round(prop_constant * angle_rad)

