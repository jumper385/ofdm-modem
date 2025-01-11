import matplotlib.pyplot as plt
import numpy as np
import random

from common.qam4_transmitter import QAM4_Transmitter
from common.qam4_receiver import QAM4_Receiver
from common.channel import awgn_channel, add_delay

# generate 32 bit random number
def random32():
    return np.random.randint(0, 2**32)

def zadoff_chu_preamble(u, N):
    """
    Generates a Zadoff-Chu preamble.
    """
    return np.exp(-1j * np.pi * u * np.arange(0, N) * (np.arange(0, N) + 1) / N)

transmitter = QAM4_Transmitter()
receiver = QAM4_Receiver()

snr_values = range(15, 50)
num_runs = 1000
error_rates = []

for snr in snr_values:
    num_errors = 0
    for _ in range(num_runs):
        data = random32()
        sig = transmitter.transmit(data)
        sig = awgn_channel(sig, snr)

        random_delay = random.randint(0, random.randint(0, 100))
        sig = add_delay(sig, random_delay)

        bits = receiver.process_received_signal(sig)

        recovered_data = 0
        for i, bit in enumerate(bits[::-1]):
            recovered_data += bit << i

        if recovered_data != data:
            num_errors += 1

    error_rate = num_errors / num_runs
    error_rates.append(error_rate)
    print(f"SNR: {snr}, Error rate: {error_rate}")

plt.plot(snr_values, error_rates)
plt.title("Error rate vs SNR")
plt.show()

# # plot the fft of the signal
# imag_sig = sig.imag
# real_sig = sig.real

# imag_freq = np.fft.rfft(imag_sig)
# real_freq = np.fft.rfft(real_sig)

# fig, ax = plt.subplots(4)
# ax[0].plot(imag_freq, label='imag')
# ax[0].plot(real_freq, label='real')
# ax[0].legend()
# ax[0].set_title('FFT of the signal')

# ax[1].plot(imag_sig)
# ax[1].plot(real_sig)
# ax[1].set_title('Signal')

# ax[2].plot(autocorrelation)
# ax[2].set_title('Autocorrelation')

# # ax[3].plot(np.sinc(2 * np.pi * 1/15 * np.linspace(-1,1, len(sig))))

# ax[3].plot(frame_fft.imag, label='imag')
# ax[3].plot(frame_fft.real, label='real')
# ax[3].legend()
# ax[3].set_title('Channel estimation')

# fig.tight_layout()

# plt.show()