import pytest
import numpy as np
import matplotlib.pyplot as plt

from common.qam_transmitter import QAMTransmitter
from common.qam_decoder import qam_decode

@pytest.mark.parametrize("qam_order, pilot_step", [
    (4, 2), 
    (16, 4), 
    (64, 2), 
    (256, 3), 
    (1024, 2), 
    (4096, 3)
])
def test_qam_transmitter(qam_order, pilot_step):
    data = np.random.randint(0, 2**32)
    transmitter = QAMTransmitter(qam_order, pilot_step)
    sig, orig_enc = transmitter.transmit(data)

    sig = np.fft.fft(sig[64:])
    sig = sig[:len(sig)//2]

    pilot_remove_mask = np.arange(len(sig)) % (pilot_step + 1) != 0
    pilot_mask = np.arange(len(sig)) % (pilot_step + 1) == 0

    pilots = sig[pilot_mask]
    max_amp = np.max(np.abs(pilots.real))
    print(max_amp)
    
    elements = sig[pilot_remove_mask]

    fig, ax = plt.subplots(3, figsize=(10, 6))
    ax[0].set_title("Original Frequency Encodings")
    ax[0].plot(orig_enc.real)
    ax[0].plot(orig_enc.imag)
    ax[0].set_xlabel("Frequency Index")
    ax[0].set_ylabel("Amplitude")

    ax[1].set_title("Transmitted Frequency Encodings")
    ax[1].plot(elements.real/max_amp)
    ax[1].plot(elements.imag/max_amp)
    ax[1].set_xlabel("Frequency Index")
    ax[1].set_ylabel("Amplitude")

    ax[2].set_title("Transmitted Frequency Encodings with Pilots")
    ax[2].plot(sig.real)
    ax[2].plot(sig.imag)
    ax[2].set_xlabel("Frequency Index")
    ax[2].set_ylabel("Amplitude")

    fig.tight_layout()
    plt.show()

    out = qam_decode(elements / max_amp, qam_order)

    assert out == data