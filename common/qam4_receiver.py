import numpy as np

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
        zadoff_chu_preamble = self.generate_zadoff_chu_sequence(2, 64)
        correlation_output = np.correlate(received_signal, zadoff_chu_preamble, mode="same")

        correlation_mean = np.mean(correlation_output)
        correlation_std = np.std(correlation_output)
        detection_threshold = correlation_mean + 3 * correlation_std
        return np.where(correlation_output > detection_threshold)
    
    def equalize_channel_response(self, ofdm_frame):
        """
        Equalizes the channel response using pilot signals.
        
        Args:
            ofdm_frame (np.array): Received OFDM frame
            
        Returns:
            np.array: Channel-equalized OFDM frame
        """
        pilot_symbols = ofdm_frame[self.pilot_subcarrier_positions]
        channel_estimate = np.mean(pilot_symbols)
        return ofdm_frame / channel_estimate
    
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

    def process_received_signal(self, received_signal):
        """
        Processes received signal to recover transmitted bits.
        
        Args:
            received_signal (np.array): Complex baseband received signal
            
        Returns:
            list: Recovered bits
        """
        frame_start_indices = self.detect_frame_start(received_signal)
        frame_start = frame_start_indices[0] + 32
        ofdm_frame = received_signal[frame_start[0]:]

        equalized_symbols = self.estimate_and_correct_channel(ofdm_frame)
        data_symbols = self.extract_data_symbols(equalized_symbols)
        recovered_bits = self.demodulate_qam4_symbols(data_symbols)
        return recovered_bits