import matplotlib.pyplot as plt
import numpy as np

# generate 32 bit random number
def random32():
    return np.random.randint(0, 2**32)

def map_to_qam4_symbol(random_number):
    """
    Maps a 32-bit integer into QAM-4 symbols by processing two bits at a time.

    00 -> 1 + 1j
    01 -> 1 - 1j
    10 -> -1 + 1j
    11 -> -1 - 1j

    The resulting symbol list is reversed and zero-padded on the left to reach a length that is a multiple of 8.
    """
    symbol_list = []

    for i in range(0, 32, 2):
        bits = (random_number >> i) & 0b11
        if bits == 0:
            symbol = 1 + 1j
        elif bits == 1:
            symbol = 1 - 1j
        elif bits == 2:
            symbol = -1 + 1j
        else:
            symbol = -1 - 1j
        symbol_list.append(symbol)
    
    symbol_list = [1 + 1j] * (8 - len(symbol_list) % 8) + symbol_list[::-1]
    return symbol_list

# allocate data subcarrier and pilot symbols 
def allocate_data_pilot_symbols(symbol_list):
    """
    Inserts pilot signals in every 4th subcarrier and data in the rest.
    Alternates between pilot signals between 1 + 1j and -1 - 1j.
    the pilots are inserted in between existing symbols

    also inserts guard bands at the beginning and end of the symbol list.

    NOTE: symbols will be padded prior to pilot insertion.
    NOTE: the symbol list will be padded until it is a multiple of 8.
    """
    symbol_list_with_pilots = []
    for i, symbol in enumerate(symbol_list):
        if i % 8 == 0:
            symbol_list_with_pilots.append(1 + 1j)
            symbol_list_with_pilots.append(symbol)
        elif i % 4 == 0:
            symbol_list_with_pilots.append(-1 - 1j)
            symbol_list_with_pilots.append(symbol)
        else:
            symbol_list_with_pilots.append(symbol)
    
    symbol_list_with_pilots = [0] * 2 + [1+1j] * 5 + symbol_list_with_pilots + [1+1j] * 8 + [0] * 2
    
    return np.array(symbol_list_with_pilots)

# generate ifft from symbol list
def generate_ifft(symbol_list):
    """
    Generates the IFFT of the symbol list.
    """
    print(symbol_list)
    return np.fft.ifft(symbol_list, 2*len(symbol_list) + 64)

# apply freuqency shaping window (hanning)
def apply_frequency_shaping_window(signal):
    """
    Applies a Hanning window to the signal.
    """
    return np.hanning(len(signal)) * signal

# apply preamble
def apply_preamble(signal):
    """
    Applies the preamble to the signal.
    """
    return np.concatenate([np.zeros(32), signal])

# apply cyclic prefix
def apply_cyclic_prefix(signal):
    """
    Applies the cyclic prefix to the signal.
    """
    return np.concatenate([signal])

# awgn channel
def awgn_channel(signal, snr_db):
    """
    Adds AWGN to the signal.
    """
    snr = 10**(snr_db / 10)
    noise_power = 1 / snr
    noise = np.sqrt(noise_power) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

symbol_list = map_to_qam4_symbol(2)
out = allocate_data_pilot_symbols(symbol_list)
sig = generate_ifft(out)
sig = apply_cyclic_prefix(sig)
# sig = apply_frequency_shaping_window(sig)
sig = apply_preamble(sig)
sig = awgn_channel(sig * 100, 10)[32:-64]

# plot the fft of the signal
imag_sig = sig.imag
real_sig = sig.real

imag_freq = np.fft.rfft(imag_sig)
real_freq = np.fft.rfft(real_sig)

fig, ax = plt.subplots(2)
ax[0].plot(imag_freq, label='imag')
ax[0].plot(real_freq, label='real')
ax[0].legend()
ax[0].set_title('FFT of the signal')
# set interval at ever 1
ax[0].set_xticks(np.arange(0, len(imag_freq), 1))
ax[0].grid()

ax[1].plot(imag_sig)
ax[1].plot(real_sig)

plt.show()