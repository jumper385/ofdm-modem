import matplotlib.pyplot as plt
import numpy as np
import random

from common.qam4_transmitter import QAM4_Transmitter
from common.channel import awgn_channel, add_delay

# generate 32 bit random number
def random32():
    return np.random.randint(0, 2**32)

def zadoff_chu_preamble(u, N):
    """
    Generates a Zadoff-Chu preamble.
    """
    return np.exp(-1j * np.pi * u * np.arange(0, N) * (np.arange(0, N) + 1) / N)

transmitter = QAM4_Transmitter()

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
    scaling = frame_fft_ref_values / pilot_positions[::5]
    x_new = np.linspace(0, 1, len(frame_fft_ref_values))
    upscale_scaling_real = np.interp(np.linspace(0, 1, len(frame_fft)), x_new, scaling.real)
    upscale_scaling_imag = np.interp(np.linspace(0, 1, len(frame_fft)), x_new, scaling.imag)
    upscale_scaling = upscale_scaling_real + 1j * upscale_scaling_imag

    return frame_fft / upscale_scaling 

def remove_pilot_symbols(frame_fft):
    """
    Removes the pilot symbols from the frame.
    """
    return np.array([frame_fft[i] for i in range(len(frame_fft)) if i % 5 != 0])

def recover_bits(frame_fft, threshold=0.5):
    """
    Recovers the data symbols from the frame.
    :param frame_fft: The frame after the FFT.
    :param threshold: The threshold for the decision.
    """
    bits = []

    for symbol in frame_fft:
        if symbol.real < threshold:
            bits.append(1)
        else:
            bits.append(0)

        if symbol.imag < threshold:
            bits.append(1)
        else:
            bits.append(0)

    return bits

snr_values = range(15, 50)
num_runs = 1000
error_rates = []

for snr in snr_values:
    num_errors = 0
    for _ in range(num_runs):
        data = random32()
        sig = transmitter.transmit(data)
        sig = awgn_channel(sig, snr)
        random_delay = random.randint(0, random.randint(0, 100))
        sig = add_delay(sig, random_delay)
        
        preamble = zadoff_chu_preamble(2, 64)
        autocorrelation = np.correlate(sig, preamble, mode="same")

        start_idx = find_preamble(sig)
        frame_start = start_idx[0] + 32
        frame = sig[frame_start[0]:]

        frame_fft = estimate_channel(frame)
        frame_fft = remove_pilot_symbols(frame_fft)
        bits = recover_bits(frame_fft)

        recovered_data = 0
        for i, bit in enumerate(bits[::-1]):
            recovered_data += bit << i

        if recovered_data != data:
            num_errors += 1

    error_rate = num_errors / num_runs
    error_rates.append(error_rate)
    print(f"SNR: {snr}, Error rate: {error_rate}")

plt.plot(snr_values, error_rates)
plt.title("Error rate vs SNR")
plt.show()

# # plot the fft of the signal
# imag_sig = sig.imag
# real_sig = sig.real

# imag_freq = np.fft.rfft(imag_sig)
# real_freq = np.fft.rfft(real_sig)

# fig, ax = plt.subplots(4)
# ax[0].plot(imag_freq, label='imag')
# ax[0].plot(real_freq, label='real')
# ax[0].legend()
# ax[0].set_title('FFT of the signal')

# ax[1].plot(imag_sig)
# ax[1].plot(real_sig)
# ax[1].set_title('Signal')

# ax[2].plot(autocorrelation)
# ax[2].set_title('Autocorrelation')

# # ax[3].plot(np.sinc(2 * np.pi * 1/15 * np.linspace(-1,1, len(sig))))

# ax[3].plot(frame_fft.imag, label='imag')
# ax[3].plot(frame_fft.real, label='real')
# ax[3].legend()
# ax[3].set_title('Channel estimation')

fig.tight_layout()

plt.show()