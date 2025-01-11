import numpy as np

def awgn_channel(signal, snr_db):
    """
    Adds AWGN to the signal.
    """
    snr = 10**(snr_db / 10)
    noise_power = 1 / snr
    noise = np.sqrt(noise_power) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

def add_delay(signal, delay):
    """
    Adds a delay to the signal.
    """
    return np.concatenate([np.zeros(delay), signal])