import pytest
import matplotlib.pyplot as plt
import numpy as np

from common.helpers import apply_noise
from common.qam_encoder import qam_encode
from common.qam_decoder import qam_decode

@pytest.mark.parametrize("qam_order", [
    4, 16, 64, 256, 1024, 4096, 16384
])
def test_decode_qam(qam_order):
    data = np.random.randint(0, 2**32)

    symbol_list = qam_encode(data, qam_order)
    sig = np.fft.ifft(symbol_list, len(symbol_list)*2)

    max_snr = 10 * np.log10(qam_order) + 30
    sig = apply_noise(sig, max_snr)

    sig = np.fft.fft(sig)[:len(sig)//2]
    out = qam_decode(sig, qam_order)

    plt.plot(sig.real)
    plt.plot(sig.imag)
    plt.show()

    assert out == data