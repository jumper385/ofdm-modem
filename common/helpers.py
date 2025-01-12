import numpy as np

def apply_noise(sig, snr):
    """
    Applies noise to the signal.
    """
    sig_power = np.mean(np.abs(sig)**2)
    noise_power = sig_power / (10**(snr/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig)) + 1j * np.random.normal(0, np.sqrt(noise_power), len(sig))
    return sig + noise