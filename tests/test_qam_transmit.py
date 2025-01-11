import pytest
import numpy as np
import matplotlib.pyplot as plt

from common.qam_transmitter import QAMTransmitter

@pytest.mark.parametrize("qam_order", [
    4, 16, 64, 256, 1024
])
def test_qam_transmitter(qam_order):
    data = np.random.randint(0, 2**32) << 32 | np.random.randint(0, 2**32)
    transmitter = QAMTransmitter(qam_order)
    sig = transmitter.transmit(data)
    print(sig.shape)

    plt.plot(sig.real)
    plt.plot(sig.imag)
    plt.show()

    # # run autocorrelation for 32 sample window
    autocorr = np.correlate(sig, sig[:32], mode='full')
    plt.plot(autocorr)
    plt.show()

    sig = np.fft.fft(sig)[:len(sig)//2]
    plt.plot(sig.real)
    plt.plot(sig.imag)
    plt.show()