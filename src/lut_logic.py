import math

def address_to_rad(address, k):
    # k-bit addresses range from 0 to 2^k - 1 so convert
    # to radians by multiplying by 2*pi / 2^k because
    # angles wrap around at 2*pi, just like address overflows
    # (wraps around) at 2^k = (1 << k)
    prop_constant = 2 * math.pi / (1 << k)

    return prop_constant * address


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
