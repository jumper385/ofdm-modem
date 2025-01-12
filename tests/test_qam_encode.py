import pytest
import matplotlib.pyplot as plt
import numpy as np

from common.helpers import apply_noise
from common.qam_encoder import encode_bits_to_symbols, normalize_symbols, qam_encode

@pytest.mark.parametrize("input_data, qam_order, symbol_length, expected", [
    (0b1100, 4, 32, [1, 1, 0, 0]),
    (0b1100, 16, 16, [3, 0]),
    (0b111000111000, 64, 8, [7, 0, 7, 0]),
    (0b1111000011110000, 256, 8, [15, 0, 15, 0]),
])
def test_encode_bits_to_symbol(input_data, qam_order, symbol_length, expected):
    out = encode_bits_to_symbols(input_data, qam_order, symbol_length)
    assert len(out) == symbol_length
    print(out)

    data_len = len(expected)
    assert np.allclose(out[-data_len:], expected)

@pytest.mark.parametrize("input_data, qam_order, symbol_length, expected", [
    (0b1100, 4, 32, [1, 1, 0, 0]),
    (0b1100, 16, 16, [3, 0]),
    (0b111000111000, 64, 8, [7, 0, 7, 0]),
    (0b1111001100001100, 256, 8, [15, 3, 0, 12]),
])
def test_normalize_symbols(input_data, qam_order, symbol_length, expected):
    symbols = encode_bits_to_symbols(input_data, qam_order, symbol_length)
    out = normalize_symbols(symbols, qam_order)
    expected_vals = np.array(expected) / (2**(np.log2(qam_order)/2) - 1)
    expected_vals = expected_vals * 2 - 1
    assert np.allclose(out[-len(expected):], expected_vals)

@pytest.mark.parametrize("qam_order", [
    4, 16, 64, 256, 1024, 4096
])
def test_encode_qam(qam_order):
    """
    Manual Inspection of QAM Constellation Diagram
    """

    out = np.array([])
    for _ in range(6000):
        val = np.random.randint(0, 2**32)
        test = qam_encode(val, qam_order, 32)
        out = np.concatenate([out, test]) 
    
    max_snr = 10 * np.log10(qam_order) + 20
    print(max_snr)
    out = apply_noise(out, max_snr)

    out_real = np.real(out)
    out_imag = np.imag(out)

    plt.figure(figsize=(5, 5))
    plt.title(f"QAM-{qam_order} Constellation Diagram")
    plt.scatter(out_real, out_imag, s=1, c='b', marker='o')
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    plt.show()

@pytest.mark.parametrize("data, qam_order, symbol_length", [
    (0b1000000110001010011101101010101, 4, 32),
    (0b10001011100101001101010101110101010101111, 16, 16),
    (0b1111010101010101111110001110001001011011, 64, 8),
    (0b10101010101000111101011110101110101111100001, 256, 8),
])
def test_ofdm_encode(data, qam_order, symbol_length):
    """
    Encodes data using OFDM
    """
    symbol_list = qam_encode(data, qam_order, symbol_length)
    sig = np.fft.ifft(symbol_list, len(symbol_list)*2)

    fft_rx = np.fft.fft(sig)[:len(sig)//2]

    fig, ax = plt.subplots(2)
    ax[0].plot(fft_rx.real)
    ax[0].plot(fft_rx.imag)
    ax[1].plot(symbol_list.real)
    ax[1].plot(symbol_list.imag)
    plt.show()