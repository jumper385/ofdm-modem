import pytest
import numpy as np
import matplotlib.pyplot as plt

from common.qam_transmitter import QAMTransmitter
from common.qam_decoder import qam_decode, preprocess_packet
from common.helpers import apply_noise 
from common.channel import free_space_path_loss, awgn_channel, add_delay, quantize_signal

@pytest.mark.parametrize("qam_order, pilot_step, symbol_length, data", [
    (16, 2, 32, 0b111111110000000011111111000000001111111100000000111111110000000),
    (64, 2, 32, 0b111111110000000011111111000000001111111100000000111111110000000),
    (256, 3, 16, 0b111111110000000011111111000000001111111100000000111111110000000),
    (1024, 2, 16, 0b111111110000000011111111000000001111111100000000111111110000000),
    (4096, 2, 16, 0b111111110000000011111111000000001111111100000000111111110000000),
    (16, 2, 32, 0b1100),
    (64, 2, 32, 0b1100),
    (256, 3, 16, 0b1100),
    (1024, 2, 16, 0b1100),
    (4096, 2, 16, 0b1100),
])
def test_qam_transmitter(qam_order, pilot_step, symbol_length, data):
    """
    Test function for QAMTransmitter.
    Verifies that the transmitted QAM signal can be decoded correctly.
    """
    # Initialize the QAM transmitter with specified parameters
    qam_transmitter = QAMTransmitter(qam_order, symbol_length, pilot_step)
    signal, original_encoded_data = qam_transmitter.transmit(data)
    
    # Calculate the maximum Signal-to-Noise Ratio (SNR) based on QAM order
    maximum_snr = 10 * np.log10(qam_order) + 15
    signal = free_space_path_loss(signal, 2000, 915e6)
    signal = apply_noise(signal, maximum_snr)
    signal = quantize_signal(signal, 12, 180)

    # decode packet
    data_elements = preprocess_packet(signal, qam_order, pilot_step)
    decoded_data = qam_decode(data_elements, qam_order)

    assert decoded_data == data