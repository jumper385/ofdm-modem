import numpy as np
import matplotlib.pyplot as plt

from common.qam_encoder import qam_encode

class QAMTransmitter():
    def __init__(self, qam_order, symbol_length, pilot_step):
        self.qam_order = qam_order
        self.pilot_steps = pilot_step
        self.cyclic_prefix_len = 32
        self.preamble_len = 15
        self.preamble_count = 1
        self.symbol_length = symbol_length
    
    def allocate_pilot_symbols(self, symbol_list):
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
            if i % int(self.pilot_steps * 2) == 0:
                symbol_list_with_pilots.append(1 + 1j)
                symbol_list_with_pilots.append(symbol)
            elif i % int(self.pilot_steps) == 0:
                symbol_list_with_pilots.append(-1 - 1j)
                symbol_list_with_pilots.append(symbol)
            else:
                symbol_list_with_pilots.append(symbol)
        
        return symbol_list_with_pilots
    
    def generate_cyclic_prefix(self, signal, cyclic_prefix_len):
        """
        Generates a cyclic prefix for the signal.
        """
        return np.concatenate([signal[-cyclic_prefix_len:], signal])
    
    def zadoff_chu_preamble(self, u, N):
        """
        Generates a Zadoff-Chu preamble.
        """
        return np.exp(-1j * np.pi * u * np.arange(0, N) * (np.arange(0, N) + 1) / N)
    
    def apply_preamble(self, signal, u, N, count):
        """
        Applies a preamble to the signal.
        """
        preamble = self.zadoff_chu_preamble(31, 32)
        sig_mean = np.mean(np.abs(signal))
        sig_std = np.std(np.abs(signal))
        preamble_amp = sig_mean + sig_std
        preamble = preamble * preamble_amp
        sig_out = np.concatenate([preamble, signal])
        sig_out = (sig_out - np.mean(sig_out)) / np.std(sig_out)

        return sig_out

    def transmit(self, data):
        """
        Transmits data using QAM modulation
        :param data: int, data to be transmitted
        """
        orig_enc = qam_encode(data, self.qam_order, self.symbol_length)
        sig = np.array(self.allocate_pilot_symbols(orig_enc))
        sig = np.fft.ifft(sig, len(sig)*2)
        sig = self.generate_cyclic_prefix(sig, self.cyclic_prefix_len)
        sig = self.apply_preamble(sig, self.preamble_len - 1, self.preamble_len, self.preamble_count)

        return sig, orig_enc