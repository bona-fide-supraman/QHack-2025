import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.wavfile import write

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hertz
CHANNELS = 1  # Mono
PERCENTAGE_TO_KEEP = 0.1  # Keep the strongest 10% of frequencies

# --- File Names ---
OUTPUT_WAV_FILENAME = "reconstructed_audio.wav"
OUTPUT_CSV_FILENAME = "sampled_fft_data.csv"

import matplotlib

matplotlib.use('TkAgg')


def full_audio_analysis_cycle():
    """
    Records audio, performs smart FFT sampling, reconstructs the signal,
    plots the results, plays the reconstructed audio, and saves the data.
    """
    # --- STAGE 1: Record the Original Signal ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    # --- STAGE 2: Calculate and Sample the FFT ---
    print("🔬 Analyzing and sampling the frequency spectrum...")
    N = len(audio_original)
    fft_full = np.fft.rfft(audio_original)
    magnitudes = np.abs(fft_full)

    threshold = np.percentile(magnitudes, 100 * (1 - PERCENTAGE_TO_KEEP))
    mask = magnitudes < threshold
    fft_sampled = fft_full.copy()
    fft_sampled[mask] = 0

    # --- STAGE 3: Reconstruct the Signal ---
    print("🏗️  Reconstructing signal from the sampled spectrum...")
    audio_reconstructed = np.fft.irfft(fft_sampled)

    # --- STAGE 4: Save and Play the Reconstructed Audio ---
    print(f"🔊 Saving and playing back the reconstructed sound...")

    # Normalize the audio to 16-bit integer format for saving
    # This is the standard format for WAV files.
    max_amplitude = np.max(np.abs(audio_reconstructed))
    if max_amplitude > 0:
        # Scale to the full range of 16-bit integers
        audio_normalized = np.int16((audio_reconstructed / max_amplitude) * 32767)
    else:
        # Handle the case of a silent recording
        audio_normalized = np.int16(audio_reconstructed)

    # Save as a .wav file
    write(OUTPUT_WAV_FILENAME, SAMPLE_RATE, audio_normalized)
    print(f"   -> Saved reconstructed audio to '{OUTPUT_WAV_FILENAME}'")

    # Play the sound
    sd.play(audio_normalized, SAMPLE_RATE)
    sd.wait()  # Wait for playback to finish
    print("   -> Playback finished.")

    # --- STAGE 5: Export FFT Data to CSV ---
    print(f"📊 Exporting sampled FFT data to CSV...")

    # Create the frequency axis for our data
    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)

    # Create a Pandas DataFrame (a professional data table)
    fft_dataframe = pd.DataFrame({
        "Frequency (Hz)": frequency_axis,
        "Magnitude": np.abs(fft_sampled),
        "Phase (radians)": np.angle(fft_sampled),
        "Real Part": fft_sampled.real,
        "Imaginary Part": fft_sampled.imag
    })

    # Save the DataFrame to a CSV file
    fft_dataframe.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"   -> Saved FFT data to '{OUTPUT_CSV_FILENAME}'")

    # --- STAGE 6: Plotting ---
    print("📈 Generating final plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Signal Reconstruction using 'Strongest Frequency' Sampling", fontsize=16)

    # Plotting code remains the same as before...
    time_axis = np.linspace(0, DURATION, num=N)
    axs[0, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[0, 0].set_title("1. Original Recorded Signal")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    axs[0, 1].plot(frequency_axis, magnitudes, color='magenta', lw=1)
    axs[0, 1].set_title("2. Full Frequency Spectrum (FFT)")
    axs[0, 1].set_xlim(0, 8000)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    axs[1, 0].plot(frequency_axis, np.abs(fft_sampled), color='yellow', lw=1)
    axs[1, 0].set_title(f"3. Sampled Spectrum (Strongest {PERCENTAGE_TO_KEEP * 100:.0f}% Kept)")
    axs[1, 0].set_xlim(0, 8000)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.4, label='Original')
    axs[1, 1].plot(time_axis, audio_reconstructed, color='red', lw=1, label='Reconstructed')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    full_audio_analysis_cycle()