import numpy as np
import matplotlib.pyplot as plt

class QAM4_Receiver():
    def __init__(self):
        self.pilot_subcarrier_positions = np.arange(0, 64, 8)
        self.pilot_subcarrier_positions = self.pilot_subcarrier_positions[2:-2]
    
    def generate_zadoff_chu_sequence(self, root_index, sequence_length):
        """
        Generates a Zadoff-Chu sequence used as a preamble for frame synchronization.
        
        Args:
            root_index (int): The root index of the Zadoff-Chu sequence
            sequence_length (int): Length of the sequence to generate
            
        Returns:
            np.array: Complex-valued Zadoff-Chu sequence
        """
        return np.exp(-1j * np.pi * root_index * np.arange(0, sequence_length) * 
                     (np.arange(0, sequence_length) + 1) / sequence_length)

    def detect_frame_start(self, received_signal):
        """
        Detects the start of a frame in the received signal using correlation with Zadoff-Chu sequence.
        
        Args:
            received_signal (np.array): Complex baseband received signal
            
        Returns:
            np.array: Indices where frame starts are detected
        """
        zadoff_chu_preamble = self.generate_zadoff_chu_sequence(63, 64)
        fwd_correlation_output = np.correlate(received_signal, zadoff_chu_preamble, mode="same")
        rev_correlation_output = np.correlate(received_signal[::-1], zadoff_chu_preamble, mode="same")
        correlation_output = rev_correlation_output[::-1] + fwd_correlation_output
        correlation_output = np.abs(correlation_output)

        correlation_mean = np.mean(correlation_output)
        correlation_std = np.std(correlation_output)

        detection_threshold = correlation_mean + 8.5 * correlation_std
        locations = np.where(correlation_output > detection_threshold)

        location_fin = []
        for i in range(1, len(locations[0])):
            if locations[0][i] - locations[0][i-1] > 64:
                location_fin.append(locations[0][i-1])

        plt.figure(figsize=(10,3))
        plt.title("Zadoff-Chu Correlation Detections")
        plt.plot(correlation_output, linewidth=0.5, color="blue", label="Signal Correlation")
        plt.axhline(correlation_mean, color="red", linestyle="--", linewidth=0.5, label="Mean")
        plt.axhline(detection_threshold, color="green", linestyle="--", linewidth=0.5, label="Threshold")
        plt.legend()
        # sig figsize to 10,3
        plt.show()

        return location_fin
    
    def equalize_channel_response(self, ofdm_frame, guard_bands=8):
        """
        Equalizes the channel response using pilot signals.
        
        Args:
            ofdm_frame (np.array): Received OFDM frame
            
        Returns:
            np.array: Channel-equalized OFDM frame
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

    
    def get_reference_pilot_symbol(self, subcarrier_index):
        """
        Returns the reference pilot symbol for a given subcarrier index.
        Pilots alternate between (1+1j) and (-1-1j) every 5 subcarriers.
        
        Args:
            subcarrier_index (int): Index of the subcarrier
            
        Returns:
            complex: Reference pilot symbol value
        """
        if subcarrier_index % 5 == 0:
            pilot_count = subcarrier_index // 5
            return (1 + 1j) if pilot_count % 2 == 0 else (-1 - 1j)
        else:
            return 1

    def pilot_value_at_subcarrier(self, k):
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
    
    def estimate_and_correct_channel(self, ofdm_frame, guard_bands=8):
        """
        Performs channel estimation and correction using pilot symbols.
        
        Args:
            ofdm_frame (np.array): Received OFDM frame
            guard_bands (int): Number of guard bands at each edge
            
        Returns:
            np.array: Channel-corrected OFDM symbols
        """
        fft_output_with_guard = np.fft.fft(ofdm_frame)[:int(len(ofdm_frame)/2)]
        fft_output = fft_output_with_guard[guard_bands:-guard_bands]

        reference_pilots = np.array([self.get_reference_pilot_symbol(k) for k in range(len(fft_output))])
        received_pilots = fft_output[::5]
        channel_response = received_pilots / reference_pilots[::5]
        
        interpolation_points = np.linspace(0, 1, len(received_pilots))
        full_range_points = np.linspace(0, 1, len(fft_output))
        
        interpolated_response_real = np.interp(full_range_points, interpolation_points, channel_response.real)
        interpolated_response_imag = np.interp(full_range_points, interpolation_points, channel_response.imag)
        interpolated_response = interpolated_response_real + 1j * interpolated_response_imag

        return fft_output / interpolated_response

    def extract_data_symbols(self, ofdm_symbols):
        """
        Extracts data symbols by removing pilot symbols from OFDM frame.
        
        Args:
            ofdm_symbols (np.array): OFDM symbols including pilots
            
        Returns:
            np.array: Data symbols only
        """
        return np.array([ofdm_symbols[i] for i in range(len(ofdm_symbols)) if i % 5 != 0])

    def demodulate_qam4_symbols(self, qam_symbols, decision_threshold=0):
        """
        Demodulates QAM4 symbols to bits using decision boundaries.
        
        Args:
            qam_symbols (np.array): Complex QAM4 symbols
            decision_threshold (float): Decision boundary threshold
            
        Returns:
            list: Demodulated bits
        """
        demodulated_bits = []

        for symbol in qam_symbols:
            demodulated_bits.append(1 if symbol.real < decision_threshold else 0)
            demodulated_bits.append(1 if symbol.imag < decision_threshold else 0)

        return demodulated_bits
    
    def bit_array_to_int(self, bit_array):
        """
        Converts a list of bits to an integer.
        
        Args:
            bit_array (list): List of bits
            
        Returns:
            int: Integer value of the bits
        """

        recovered_data = 0
        for i, bit in enumerate(bit_array[::-1]):
            recovered_data += bit << i
        
        return recovered_data

    def process_received_signal(self, received_signal):
        """
        Processes received signal to recover transmitted data by:
        1. Detecting frame start positions using Zadoff-Chu correlation
        2. Extracting individual OFDM frames
        3. Equalizing channel response
        4. Extracting data symbols (removing pilots)
        5. Demodulating QAM4 symbols to bits
        6. Converting bit sequences to integers
        
        Args:
            received_signal (np.array): Complex baseband received signal samples
            
        Returns:
            list: List of recovered integer values from each OFDM frame
        """
        frame_start_positions = self.detect_frame_start(received_signal)
        recovered_integers = []
        
        for frame_position in frame_start_positions:
            # Skip preamble length (32) and extract frame of length 92
            ofdm_frame_start = frame_position + 32 + 32
            ofdm_frame_end = ofdm_frame_start + 92
            current_frame = received_signal[ofdm_frame_start:ofdm_frame_end]

            if len(current_frame) < 92:
                continue
            else:
                channel_corrected_symbols = self.estimate_and_correct_channel(current_frame)

            if len(channel_corrected_symbols) == 30:
                payload_symbols = self.extract_data_symbols(channel_corrected_symbols)
                demodulated_bits = self.demodulate_qam4_symbols(payload_symbols)
                frame_integer = self.bit_array_to_int(demodulated_bits)
                recovered_integers.append(frame_integer)

                fig, ax = plt.subplots(2, figsize=(10, 6))
                # make lines transparent
                ax[0].plot(received_signal.real, color='red', linewidth=0.5, label="Real")  
                ax[0].plot(received_signal.imag, color='lightgray', linewidth=0.5, label="Imaginary")
                ax[0].axvline(ofdm_frame_start, color="red", linestyle="--", linewidth=1, label="Frame Start")
                ax[0].axvline(ofdm_frame_end, color="red", linestyle="--", linewidth=1, label="Frame End")
                ax[0].axvline(frame_position, color="green", linestyle="--", linewidth=1, label="Frame Detection")
                ax[0].legend()
                ax[0].set_title("Received Signal")

                ax[1].set_title("Recovered Payload Symbols (FFT)")
                ax[1].plot(payload_symbols.real, label="Real", linewidth=0.5, color='red')
                ax[1].plot(payload_symbols.imag, label="Imaginary", linewidth=0.5, color='black')

                fig.tight_layout()
                plt.show()
            
            else:
                print("Not enough data symbols in frame. Skipping...")

        return recovered_integers