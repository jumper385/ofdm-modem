import numpy as np

class QAM4_Transmitter():
    def __init__(self):
        self.pilot_symbols = [1 + 1j, -1 - 1j]
        self.guard_band = [0] * 2
        self.pilot_interval = 4
        self.pilot_symbol_interval = 8

        self.data_len = 32
        self.cycling_prefix_len = 32

    def map_to_qam4_symbol(self, input_data):
        """
        Maps a 32-bit integer into QAM-4 symbols by processing two bits at a time.

        00 -> 1 + 1j
        01 -> 1 - 1j
        10 -> -1 + 1j
        11 -> -1 - 1j

        The resulting symbol list is reversed and zero-padded on the left to reach a length that is a multiple of 8.
        """
        symbol_list = []

        for i in range(0, self.data_len, 2):
            bits = (input_data >> i) & 0b11
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
    
    def allocate_data_pilot_symbols(self, symbol_list):
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
            if i % self.pilot_symbol_interval == 0:
                symbol_list_with_pilots.append(1 + 1j)
                symbol_list_with_pilots.append(symbol)
            elif i % self.pilot_interval == 0:
                symbol_list_with_pilots.append(-1 - 1j)
                symbol_list_with_pilots.append(symbol)
            else:
                symbol_list_with_pilots.append(symbol)
        
        
        symbol_list_with_pilots = self.guard_band + [1+1j] * 6 + symbol_list_with_pilots + [1+1j] * 6 + self.guard_band
        return symbol_list_with_pilots

    def generate_ifft(self, symbol_list):
        """
        Generates the IFFT of the symbol list.
        """
        return np.fft.ifft(symbol_list, len(symbol_list)*2)
    
    def apply_frequency_shaping(self, signal):
        """
        Applies a frequency shaping window to the IFFT output.
        """
        window = np.sinc(2 * np.pi * 1/48 * np.linspace(-1, 1, len(signal)))
        return window * signal   
    
    def _zadoff_chu_preamble(self, u, N):
        """
        Generates a Zadoff-Chu preamble.
        """
        return np.exp(-1j * np.pi * u * np.arange(0, N) * (np.arange(0, N) + 1) / N)

    def apply_preamble(self, signal):
        """
        Applies a preamble to the signal.
        """
        preamble = self._zadoff_chu_preamble(63, 64)
        sig_mean = np.mean(np.abs(signal))
        sig_std = np.std(np.abs(signal))
        preamble_amp = sig_mean + sig_std
        preamble = preamble * preamble_amp
        return np.concatenate([preamble, signal])
    
    def add_cyclic_prefix(self, signal):
        """
        Adds a cyclic prefix to the signal.
        """
        return np.concatenate([signal[-self.cycling_prefix_len:], signal])
    
    def transmit(self, input_data):
        """
        Transmits a 32-bit integer input data.
        """
        sig = self.map_to_qam4_symbol(input_data)
        sig = self.allocate_data_pilot_symbols(sig)
        sig = self.generate_ifft(sig)
        sig = self.add_cyclic_prefix(sig)
        sig = self.apply_preamble(sig)
        sig = self.apply_frequency_shaping(sig)
        return sig