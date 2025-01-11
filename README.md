# OFDM Modem Design
This project involves the design and implementation of an Orthogonal Frequency-Division Multiplexing (OFDM) modem. It focuses on signal processing techniques, hardware integration, and performance optimization to ensure efficient and reliable data transmission.

# Background
OFDM is a widely used modulation scheme in modern communication systems due to its robustness against frequency-selective fading channels. It divides the available bandwidth into multiple orthogonal subcarriers, allowing for efficient data transmission and reception. This project aims to develop an OFDM modem that can transmit and receive data over a wireless channel, incorporating signal processing algorithms and hardware components for real-time operation.

# Features
- **Pluto SDR Integration**: Utilize the Analog Devices ADI Pluto SDR for signal transmission and reception.
- **QAM4 OFDM Implementation**: Implement a basic OFDM transmitter and receiver with pilot symbols for channel estimation.
- **OFDM Signal Processing Techniques**: Incorporate Zadoff-Chu preamble, channel estimation, and cyclic prefix for robust signal reception.

# Installation
1. Clone the repository:
```bash
git clone git@github.com:jumper385/ofdm-modem.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Connect the ADI Pluto SDR to the host machine and configure the device settings.
4. Find the Pluto SDR device address and update the configuration in `main.py`.
```bash
iio_info -S usb
```
5. Run the main script to start the OFDM modem simulation:
```bash
python main.py
```
