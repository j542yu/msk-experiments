from dataclasses import dataclass

"""
Since MSK is a type of FSK, the amplitude is always between
1 and -1. For convenience, let modulator output be n-bit integer
code representing some value between -1 to 1. Since there must be a
code for both endpoints, each LSB represents (1 - (-1))/(2^n - 1) = 2/(2^n - 1)
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
 m samples/  ┌────────────────────┐        │  ┌───────────┐  │                                           
  symbol     │ 0 -> phi_if - pi/2 │        ▼  │phase      │  │            ┌───────────┐        ┌────────┐
─────/──────►│ 1 -> phi_if + pi/2 ├─────/────►│accumulator┼─►┴──────/────►│ cos LUT   ├──/────►│  DAC   │
     1       └────────────────────┘    k+j    └───────────┘         k     └───────────┘  n     └────────┘

(phi_if is the constant phase increase from IF)
Made with https://asciiflow.com/#/     
"""

@dataclass
class ModulationParams:
    # Bit widths
    n: int = 6
    k: int = 8
    j: int = 2
    m: int = 4

    # Signal
    num_bits: int = 1000
    seed: int = 10086 # for generating bitstream

    # Sampling rate (bottle-necked by accumulator speed?)
    f_s: int = 1_000_000  # Hz

    """
    Intermediate frequency (IF)

    For MSK, the two frequencies are
    f_0 = f_if - 1/(4 * T) = f_if - f_s/(4 * m)
    f_1 = f_if + f_s/(4 * m)
    (where period T = m/f_s since m samples per symbol)

    Since f_0, f_1 must be positive and because of Nyquist theorem,
    0 < f_0 < f_s/2 and 0 < f_1 < f_s/2

    f_0 inequality gives f_s/(4 * m) < f_if < f_s/2 + f_s/(4 * m)
    f_1 inequality gives -f_s/(4 * m) < f_if < f_s/2 - f_s/(4 * m)

    Together, f_s/(4 * m) < f_if < f_s/2 - f_s/(4 * m)

    For f_s = 1MHz (picked arbitrary value, need to see simulation results)
    31.25kHz < f_if < 468.75 kHz

    https://en.wikipedia.org/wiki/Intermediate_frequency
    Since a higher IF means less samples per carrier cycle, that makes the
    output waveform more distorted. Thus, pick something low.
    """
    f_if: int = 40_000 # Hz
