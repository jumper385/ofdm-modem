import numpy as np

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