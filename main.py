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
    
    
    symbol_list_with_pilots = [0] * 2 + [1+1j] * 6 + symbol_list_with_pilots + [1+1j] * 6 + [0] * 2
    
    return np.array(symbol_list_with_pilots)

# generate ifft from symbol list
def generate_ifft(symbol_list):
    """
    Generates the IFFT of the symbol list.
    """
    return np.fft.ifft(symbol_list, 2*len(symbol_list))

# apply freuqency shaping window (hanning)
def apply_frequency_shaping_window(signal):
    """
    Applies a Hanning window to the signal.
    """
    return np.hanning(len(signal)) * signal

# apply preamble
def zadoff_chu_preamble(u, N):
    """
    Generates a Zadoff-Chu preamble.
    """
    return np.exp(-1j * np.pi * u * np.arange(0, N) * (np.arange(0, N) + 1) / N)

def apply_preamble(signal, u, N):
    """
    Applies the Zadoff-Chu Preamble
    """
    preamble = zadoff_chu_preamble(u, N)
    return np.concatenate([preamble, signal])

# apply cyclic prefix
def apply_cyclic_prefix(signal):
    """
    Applies the cyclic prefix to the signal.
    """
    return np.concatenate([signal])

############# Channel Model #############

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

############# Receiver #############

def find_preamble(signal):
    """
    Finds the preamble in the signal.
    """
    preamble = zadoff_chu_preamble(2, 64)
    correlation = np.correlate(signal, preamble, mode="same")

    mean = np.mean(correlation)
    std = np.std(correlation)
    threshold = mean + 3 * std
    return np.where(correlation > threshold)

def equalize_frame(frame):
    """
    uses the positions of the pilot signals to estimate the channel and equalize the frame.
    this reverses the effects of allocate_data_pilot_symbols 
    """
    pilot_positions = np.arange(0, len(frame), 8)
    pilot_positions = pilot_positions[2:-2]
    pilot_values = frame[pilot_positions]

    # estimate the channel
    channel = np.mean(pilot_values)
    return frame / channel

def pilot_value_at_subcarrier(k):
    """
    Return the known pilot value at subcarrier k.
    Pattern: pilot on multiples of 4, alternating between (1+1j) and (-1-1j).
    """
    if k % 5 == 0:
        # Count how many pilot positions we’ve had
        pilot_number = k // 5
        if pilot_number % 2 == 0:
            return 1 + 1j
        else:
            return -1 - 1j
    else:
        # Not a pilot subcarrier
        return 1

def estimate_channel(frame, guard_bands=8):
    """
    Estimates the channel from the pilot values in the frame.
    """
    frame_fft_with_guard = np.fft.fft(frame)[:int(len(frame)/2)]
    frame_fft = frame_fft_with_guard[guard_bands:-guard_bands]

    # Find the pilot positions
    pilot_positions = np.array([pilot_value_at_subcarrier(k) for k in range(len(frame_fft))])
    # get actual values in the frame at pilot signals
    frame_fft_ref_values = frame_fft[::5]
    print(frame_fft_ref_values)
    print(pilot_positions[::5])
    scaling = frame_fft_ref_values / pilot_positions[::5]
    x_new = np.linspace(0, 1, len(frame_fft_ref_values))
    print(len(scaling))
    print(len(x_new))
    upscale_scaling_real = np.interp(np.linspace(0, 1, len(frame_fft)), x_new, scaling.real)
    upscale_scaling_imag = np.interp(np.linspace(0, 1, len(frame_fft)), x_new, scaling.imag)
    upscale_scaling = upscale_scaling_real + 1j * upscale_scaling_imag

    return frame_fft / upscale_scaling 

symbol_list = map_to_qam4_symbol(0b1100110111011011)
# symbol_list = np.zeros(len(symbol_list))
out = allocate_data_pilot_symbols(symbol_list)
print(out)
sig = generate_ifft(out)
# make sig = 0 for now
# sig = apply_cyclic_prefix(sig)
# sig = apply_frequency_shaping_window(sig)
sig = apply_preamble(sig, 2, 64)
# sig = apply_frequency_shaping_window(sig)
sig = awgn_channel(sig, 36)
sig = add_delay(sig, 10)

preamble = zadoff_chu_preamble(2, 64)
autocorrelation = np.correlate(sig, preamble, mode="same")

start_idx = find_preamble(sig)
frame_start = start_idx[0] + 32
print(frame_start)
frame = sig[frame_start[0]:]

frame_fft = estimate_channel(frame)

# plot the fft of the signal
imag_sig = frame.imag
real_sig = frame.real

imag_freq = np.fft.rfft(imag_sig)
real_freq = np.fft.rfft(real_sig)

fig, ax = plt.subplots(4)
ax[0].plot(imag_freq, label='imag')
ax[0].plot(real_freq, label='real')
ax[0].legend()
ax[0].set_title('FFT of the signal')

ax[1].plot(imag_sig)
ax[1].plot(real_sig)
ax[1].set_title('Signal')

ax[2].plot(autocorrelation)
ax[2].set_title('Autocorrelation')

ax[3].plot(frame_fft.imag, label='imag')
ax[3].plot(frame_fft.real, label='real')
ax[3].legend()
ax[3].set_title('Channel estimation')

fig.tight_layout()

plt.show()