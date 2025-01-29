import numpy as np

def preprocess_packet(data, qam_order, pilot_step):
    signal_fft = np.fft.fft(data[64:])
    signal_fft = signal_fft[:len(signal_fft)//2]

    data_mask = np.arange(len(signal_fft)) % (pilot_step + 1) != 0
    pilot_mask = np.arange(len(signal_fft)) % (pilot_step + 1) == 0

    pilot_signals = signal_fft[pilot_mask]
    max_real_amplitude = np.max(np.abs(pilot_signals.real))

    data_elements = signal_fft[data_mask]
    data_elements = data_elements / max_real_amplitude
    return data_elements

def qam_decode(data, qam_order):
    """
    Decodes QAM symbols into bits
    :param data: list, list of QAM symbols
    :param qam_order: int, QAM order
    """
    qam_base = int(np.log2(qam_order)/2)
    qam_base = int(2**qam_base)

    real_bins = np.linspace(-1, 1, qam_base)
    imag_bins = np.linspace(-1, 1, qam_base)

    find_closest = lambda x, bins: np.argmin(np.abs(bins - x))
    decoded_data = 0
    shift_size =  int(np.log2(qam_base))
    for i in range(0, len(data)):
        real_idx = find_closest(data[i].real, real_bins)
        imag_idx = find_closest(data[i].imag, imag_bins)

        decoded_data = decoded_data << shift_size | real_idx
        decoded_data = decoded_data << shift_size | imag_idx
    
    return decoded_data