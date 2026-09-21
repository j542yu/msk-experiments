import numpy as np
import matplotlib.pyplot as plt
import modulate, demodulate

params = modulate.SimulationParams(
    num_bits=10000,
    samples_per_symbol=1,
    modulation="MSK",
    EbN0_dB=15,
    freq_offset=0.05,
    timing_offset=0.20,
    quant_bits=3,
)

tx = modulate.generate_test_signal(params)

rx = demodulate.gmsk_msk_demodulator(
    tx["signal"],
    samples_per_symbol=params.samples_per_symbol,
    filter_BT=0.5,
)

transmitted_bits = tx["bits"][1:]
detected_bits = rx["detected_bits"]


def measure_ebn0(signal, noisy_signal, samples_per_symbol):

    noise = noisy_signal - signal

    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)

    Eb = signal_power * samples_per_symbol

    # Complex noise power = N0
    N0 = noise_power

    measured_EbN0 = Eb / N0

    return 10 * np.log10(measured_EbN0)

noisy = modulate.add_awgn(
    tx["signal"],
    6,
    params.samples_per_symbol,
    np.random.default_rng(params.seed)
)

print(
    measure_ebn0(
        tx["signal"],
        noisy,
        params.samples_per_symbol
    )
)

# print(detected_bits)
# print(transmitted_bits)

# BER = np.mean(transmitted_bits != detected_bits)

# print(BER)

# plt.figure()

# plt.plot(
#     rx["decision_metric"][:100],
#     ".-"
# )

# plt.axhline(0, linestyle="--")

# plt.xlabel("Symbol")
# plt.ylabel("Differential decision metric")
# plt.title("Differential Detector Output")
# plt.grid()

# plt.show()