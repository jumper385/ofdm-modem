## 0.2.0 (2025-01-12)

### Feat

- Add QAMTransmitter class for QAM modulation and corresponding tests
- Implement QAM encoding and decoding functions with corresponding tests

### Fix

- Update QAM4 receiver and transmitter to enhance signal processing and visualization
- Update test_qam_transmitter to include new data and adjust SNR calculation
- Update QAMTransmitter to accept symbol length and modify tests accordingly
- Remove unnecessary print statement from test_qam_transmitter
- Enhance QAMTransmitter to support variable pilot steps and update tests for new functionality
- Update test for qam_decode to include symbol length in encoding
- Enhance QAM encoding functions to support target symbol length
- Add noise application function to signal processing helper

## 0.1.1 (2025-01-11)

### Fix

- Update Zadoff-Chu preamble length and enhance plotting in QAM4 receiver; improve channel equalization method

## 0.1.0 (2025-01-11)

### Feat

- Integrate ADI Pluto SDR for signal transmission and reception; update processing flow in main.py
- Adjust SNR range and increase simulation runs; introduce random delay in signal processing
- Add functions to remove pilot symbols and recover bits from the frame, and implement error rate analysis over varying SNR values
- Enhance signal processing with Zadoff-Chu preamble and channel estimation functions
- Implement QAM-4 symbol mapping and signal processing pipeline with AWGN channel simulation

### Fix

- Reduce number of runs in transmission simulation; add random delay to transmitted packets
- Refine frame start detection in QAM4 receiver; optimize location filtering and remove unused plotting code
- Adjust detection threshold in QAM4 receiver and optimize transmission parameters in main.py
- Improve QAM4 receiver correlation and frame processing; add cyclic prefix in transmitter
- Enhance QAM4 receiver with integer recovery and improve signal processing in main.py
- Comment out redundant layout adjustment and plot display in main.py

### Refactor

- Add QAM4 receiver class for signal processing and bit recovery; integrate into main simulation
- Implement AWGN channel and delay functions; refactor main.py to use new channel module
- Add QAM-4 transmitter class and integrate into main simulation; update .gitignore
