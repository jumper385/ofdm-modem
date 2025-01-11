import numpy as np

def encode_bits_to_symbols(input_data, qam_order):
    """
    Encodes input data into QAM symbols
    :param input_data: int, input data to be encoded
    :param qam_order: int, QAM order
    """
    bits_per_symbol = int(np.log2(qam_order)/2) 
    symbols = []
    for i in range(0, len(bin(input_data)) - 2, bits_per_symbol):
        bits = (input_data >> i) & (2 ** bits_per_symbol - 1)
        symbols.append(bits)
    
    return np.asarray(symbols[::-1])

def normalize_symbols(symbols, qam_order):
    """
    Normalizes the symbols to be between -1 and 1
    :param symbols: list, list of symbols
    :param qam_order: int, QAM order
    """
    qam_base = np.log2(qam_order)/2 
    normalized_symbols = np.array(symbols) / (2**qam_base - 1)
    return normalized_symbols * 2 - 1

def reshape_to_complex(normalized_symbols):
    normalized_symbols = normalized_symbols.reshape(-1, 2)
    normalized_symbols = normalized_symbols[:, 0] + 1j * normalized_symbols[:, 1]
    return normalized_symbols

def qam_encode(input_data, qam_order):
    """
    Encodes input data into QAM symbols by 
    1. breaking bits into groupings of bits_per_symbol
    2. generating positional vector w.r.t to maximum qam_order
    3. assembles the vector into a list of complex numbers for QAM symbols

    :param input_data: int, input data to be encoded
    :param qam_order: int, QAM order
    """
    symbols = encode_bits_to_symbols(input_data, qam_order)
    if len(symbols) % 2 != 0:
        symbols = np.concatenate([[0], symbols])

    normalized_symbols = normalize_symbols(symbols, qam_order)
    complex_symbol = reshape_to_complex(normalized_symbols)

    return complex_symbol