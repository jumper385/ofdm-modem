import numpy as np

def free_space_path_loss(signal, distance_m, sig_freq_hz):
    """
    Calculates the free space path loss. based on the friis equation.
    """
    speed_of_light = 3e8
    wavelength = speed_of_light / sig_freq_hz
    return signal / (4 * np.pi * distance_m / wavelength)**2

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

def quantize_signal(signal, adc_bit_depth, amp_gain_db):
    """
    Quantizes the signal based on the ADC bit depth.
    """
    quantization_levels = 2**adc_bit_depth
    sig_amp = signal * 10**(amp_gain_db/20)

    sig_power = np.mean(np.abs(sig_amp)**2)

    sig_real = np.round(np.real(sig_amp) * quantization_levels) / quantization_levels
    sig_imag = np.round(np.imag(sig_amp) * quantization_levels) / quantization_levels
    signal_quantized = sig_real + 1j * sig_imag

    return signal_quantized