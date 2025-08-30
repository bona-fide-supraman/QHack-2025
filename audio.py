import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hertz (samples per second)
CHANNELS = 1  # Mono audio

# To prevent the PyCharm backend bug, we can force Matplotlib to use a reliable one.
# This is a good safety measure.
import matplotlib

matplotlib.use('TkAgg')


def record_and_analyze():
    """
    Records audio for a fixed duration, then plots its time-domain waveform
    and frequency-domain spectrum.
    """
    # 1. --- Record Audio ---
    print(f"🎙️  Recording for {DURATION} seconds... Speak into your microphone.")

    # sd.rec() records audio and returns it as a NumPy array.
    # We add a sd.wait() call to ensure the recording is finished before proceeding.
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()  # Wait until the recording is finished

    print("✅ Recording complete. Analyzing and plotting...")

    # The recorded data is a 2D array (frames x channels), so we flatten it
    # for a mono signal.
    audio_data = audio_data.flatten()

    # 2. --- Plot the Sound Wave (Time Domain) ---

    # Create a time axis in seconds for the x-axis of our plot
    time_axis = np.linspace(0, DURATION, num=len(audio_data))

    # Create a figure with two subplots, arranged vertically
    plt.figure(figsize=(10, 8))

    # First subplot: The sound wave
    plt.subplot(2, 1, 1)  # (2 rows, 1 column, select the 1st plot)
    plt.plot(time_axis, audio_data, color='cyan', lw=1)
    plt.title("Sound Wave (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 3. --- Calculate and Plot the Fourier Transform (Frequency Domain) ---

    # Perform the Fast Fourier Transform (FFT)
    # np.fft.rfft is used for real-valued inputs like our audio signal.
    N = len(audio_data)
    fft_data = np.fft.rfft(audio_data)

    # The result of the FFT is complex numbers. We take the absolute value to get the magnitude.
    fft_magnitude = np.abs(fft_data)

    # Create the frequency axis for the x-axis of our plot
    # np.fft.rfftfreq gives us the correct frequency bins for our rfft output.
    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)

    # Second subplot: The frequency spectrum
    plt.subplot(2, 1, 2)  # (2 rows, 1 column, select the 2nd plot)
    plt.plot(frequency_axis, fft_magnitude, color='magenta', lw=1)
    plt.title("Frequency Spectrum (Fourier Transform)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Optional: Limit the x-axis to a reasonable range for human voice/most sounds
    plt.xlim(0, 5000)

    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout()

    # Show the final plot window
    plt.show()


if __name__ == '__main__':
    record_and_analyze()