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

        # chain together random multiple signals
        sig = np.array([])
        data_hist = []
        for _ in range(10):
            data = random32()
            packet = transmitter.transmit(data)
            random_delay = random.randint(0, random.randint(0, 100))
            packet = add_delay(packet, random_delay)
            sig = np.concatenate([sig, packet])
            data_hist.append(data)

        # pass through channel
        sig = awgn_channel(sig, snr)

        # recover bits from chennel
        bits = receiver.process_received_signal(sig)

        # count number of errors 
        num_errors += sum([bits[i] != data_hist[i] for i in range(len(bits))])

    error_rate = num_errors / (num_runs * 10)
    error_rates.append(error_rate)
    print(f"SNR: {snr}, Error rate: {error_rate}")

plt.plot(snr_values, error_rates)
plt.title("Error rate vs SNR")
plt.show()