import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.wavfile import write

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 4000  # Hertz
CHANNELS = 1  # Mono

# --- The number of data points to use for BOTH reconstruction AND the final CSV ---
NUM_DATA_POINTS = 10000

# --- File Names ---
OUTPUT_WAV_FILENAME = f"reconstructed_from_{NUM_DATA_POINTS}_points.wav"
OUTPUT_CSV_FILENAME = f"fft_data_{NUM_DATA_POINTS}_points.csv"

import matplotlib

matplotlib.use('TkAgg')


def extreme_reconstruction_cycle():
    """
    Records audio, reduces its FFT to a tiny number of key points,
    and then reconstructs the audio FROM ONLY those few points.
    """
    # --- STAGE 1 & 2: Record and get the full FFT ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    print("🔬 Analyzing frequency spectrum...")
    N = len(audio_original)
    fft_full = np.fft.rfft(audio_original)

    # --- STAGE 3: Create the HIGHLY reduced FFT data ---
    print(f"✂️  Reducing spectrum to only the {NUM_DATA_POINTS} strongest frequency peaks...")

    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)
    full_magnitudes = np.abs(fft_full)

    num_full_points = len(frequency_axis)
    bin_size = num_full_points // NUM_DATA_POINTS

    peak_indices = []
    for i in range(NUM_DATA_POINTS):
        start_index = i * bin_size
        end_index = start_index + bin_size
        mag_slice = full_magnitudes[start_index:end_index]
        if len(mag_slice) > 0:
            max_index_in_slice = np.argmax(mag_slice)
            peak_indices.append(start_index + max_index_in_slice)

    fft_from_reduced = np.zeros_like(fft_full)
    for index in peak_indices:
        fft_from_reduced[index] = fft_full[index]

    # --- STAGE 4: Reconstruct Signal FROM THE REDUCED DATA ---
    print("🏗️  Reconstructing signal from the reduced data...")
    audio_reconstructed = np.fft.irfft(fft_from_reduced)

    # --- STAGE 5: Save and Play the Reconstructed Audio ---
    print(f"🔊 Saving and playing back the sound reconstructed from only {NUM_DATA_POINTS} points...")
    max_amplitude = np.max(np.abs(audio_reconstructed))
    if max_amplitude > 0:
        audio_normalized = np.int16((audio_reconstructed / max_amplitude) * 32767)
    else:
        audio_normalized = np.int16(audio_reconstructed)

    write(OUTPUT_WAV_FILENAME, SAMPLE_RATE, audio_normalized)
    print(f"   -> Saved reconstructed audio to '{OUTPUT_WAV_FILENAME}'")
    sd.play(audio_normalized, SAMPLE_RATE)
    sd.wait()
    print("   -> Playback finished.")

    # --- STAGE 6: Export the reduced data to CSV (with rounding) ---
    print(f"📊 Exporting {NUM_DATA_POINTS} data points to CSV...")

    reduced_freqs = frequency_axis[peak_indices]
    reduced_mags = full_magnitudes[peak_indices]
    reduced_complex = fft_full[peak_indices]

    fft_dataframe_reduced = pd.DataFrame({
        "Frequency (Hz)": reduced_freqs,
        "Magnitude": reduced_mags,
        "Real Part": reduced_complex.real,
        "Imaginary Part": reduced_complex.imag
    })

    fft_dataframe_reduced = fft_dataframe_reduced.round(2)
    fft_dataframe_reduced.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"   -> Saved {len(fft_dataframe_reduced)} data points to '{OUTPUT_CSV_FILENAME}'")

    # --- STAGE 7: Plotting ---
    print("📈 Generating final plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Signal Reconstruction from only {NUM_DATA_POINTS} Strongest Frequencies", fontsize=16)

    # --- THIS LINE WAS MISSING. IT IS THE FIX. ---
    time_axis = np.linspace(0, DURATION, num=N)

    axs[0, 0].plot(frequency_axis, full_magnitudes, color='magenta', lw=1)
    axs[0, 0].set_title("1. Full Frequency Spectrum")
    axs[0, 0].set_xlim(0, 8000)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    axs[0, 1].plot(frequency_axis, np.abs(fft_from_reduced), color='yellow', lw=1)
    axs[0, 1].set_title(f"2. Reduced Spectrum ({NUM_DATA_POINTS} Points Kept)")
    axs[0, 1].set_xlim(0, 8000)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    axs[1, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[1, 0].set_title("3. Original Recorded Signal")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.3, label='Original')
    axs[1, 1].plot(time_axis, audio_reconstructed, color='red', lw=1,
                   label=f'Reconstructed from {NUM_DATA_POINTS} points')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    extreme_reconstruction_cycle()