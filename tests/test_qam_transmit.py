import pytest
import numpy as np
import matplotlib.pyplot as plt

from common.qam_transmitter import QAMTransmitter
from common.qam_decoder import qam_decode
from common.helpers import apply_noise

@pytest.mark.parametrize("qam_order, pilot_step, symbol_length", [
    (16, 2, 32),
    (64, 2, 32),
    (256, 3, 16), 
    (1024, 2, 16), 
])
def test_qam_transmitter(qam_order, pilot_step, symbol_length):
    data = 0b10010101011110010101001010010101001010101010111101001
    print(data)
    transmitter = QAMTransmitter(qam_order, symbol_length, pilot_step)
    sig, orig_enc = transmitter.transmit(data)
    sig = apply_noise(sig, 10 * np.log10(qam_order) + 10)

    sig = np.fft.fft(sig[64:])
    sig = sig[:len(sig)//2]

    pilot_remove_mask = np.arange(len(sig)) % (pilot_step + 1) != 0
    pilot_mask = np.arange(len(sig)) % (pilot_step + 1) == 0

    pilots = sig[pilot_mask]
    max_amp = np.max(np.abs(pilots.real))
    
    elements = sig[pilot_remove_mask]

    fig, ax = plt.subplots(3, figsize=(10, 6))
    ax[0].set_title(f"Original Frequency Encodings with QAM-{qam_order}")
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