import matplotlib.pyplot as plt
import numpy as np
import random

import adi

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

sdr = adi.Pluto("usb:0.3.5")
sdr.rx_rf_bandwidth = int(40e6)
sdr.tx_rf_bandwidth = int(40e6)
sdr.rx_lo = int(915e6)
sdr.tx_lo = int(915e6)
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 30

snr_values = range(15, 50)
num_runs = 10
error_rates = []

data = random32()

tx_sig = transmitter.transmit(data)
tx_sig = add_delay(tx_sig, random.randint(0,100))

sdr.tx_destroy_buffer()
sdr.tx_cyclic_buffer = True
sdr.rx_destroy_buffer()
sdr.tx(2**12 * tx_sig)

sdr.rx_destroy_buffer()
sdr.rx_buffer_size = len(tx_sig) * 10

for _ in range(10):
    rx_sig = sdr.rx()

rx_bits = receiver.process_received_signal(rx_sig)
print(rx_bits)
print(data)