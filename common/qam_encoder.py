import numpy as np

def encode_bits_to_symbols(input_data, qam_order, symbol_length):
    """
    encodes an input data up to the target symbol length. arranges the computed symbols lsb first
    :param input_data: int, input data to be encoded
    :param qam_order: int, QAM order
    :param symbol_length: int, even numbered target symbol length
    :return: np.array, list of symbols
    """
    if symbol_length % 2 != 0:
        raise ValueError("symbol length must be even")

    bits_per_symbol = int(np.log2(qam_order)/2) 
    symbols = []

    for i in range(0, len(bin(input_data)) - 2, bits_per_symbol):
        bits = (input_data >> i) & (2 ** bits_per_symbol - 1)
        symbols.append(bits)
    
    if len(symbols) > symbol_length:
        raise ValueError("input data symbols exceed maximum symbol length")
    
    curr_length = len(symbols)
    if curr_length < symbol_length:
        symbols += [0] * (symbol_length - curr_length)
    
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

def qam_encode(input_data, qam_order, symbol_length):
    """
    Encodes the input data to QAM symbols of the target symbol length and QAM order
    :param input_data: int, input data to be encoded
    :param qam_order: int, QAM order
    :param symbol_length: int, target symbol length

    :return: np.array, list of complex QAM symbols
    """
    symbols = encode_bits_to_symbols(input_data, qam_order, symbol_length * 2)

    normalized_symbols = normalize_symbols(symbols, qam_order)
    complex_symbol = reshape_to_complex(normalized_symbols)

    return complex_symbol