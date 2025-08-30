import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io.wavfile import write

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 4000  # Back to a higher quality sample rate
CHANNELS = 1  # Mono

# --- NEW CONFIGURATION: The exact number of top frequencies to keep ---
NUM_TOP_FREQUENCIES = 25

# --- File Names ---
OUTPUT_WAV_FILENAME = f"reconstructed_from_top_{NUM_TOP_FREQUENCIES}_freqs.wav"
OUTPUT_CSV_FILENAME = f"fft_data_top_{NUM_TOP_FREQUENCIES}_freqs.csv"

import matplotlib

matplotlib.use('TkAgg')


def top_n_reconstruction_cycle():
    """
    Records audio, finds the N absolute strongest frequency components,
    and reconstructs the audio from only those top N components.
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

    # --- STAGE 3 (REVISED): Find the Top N Strongest Frequencies ---
    print(f"✂️  Finding the {NUM_TOP_FREQUENCIES} absolute strongest frequency peaks...")

    full_magnitudes = np.abs(fft_full)

    # Use np.argsort to get the indices of the magnitudes in ascending order.
    # The indices of the strongest frequencies will be at the end of this array.
    sorted_indices = np.argsort(full_magnitudes)

    # Get the indices of the top N frequencies by taking the last N elements.
    top_n_indices = sorted_indices[-NUM_TOP_FREQUENCIES:]

    # --- Create the new, extremely sparse FFT array for reconstruction ---
    # Start with an array of all zeros
    fft_from_top_n = np.zeros_like(fft_full)
    # Now, copy ONLY the peak values from the full FFT into our new sparse array
    fft_from_top_n[top_n_indices] = fft_full[top_n_indices]

    # --- STAGE 4: Reconstruct Signal FROM THE TOP N DATA ---
    print("🏗️  Reconstructing signal from the top N frequencies...")
    audio_reconstructed = np.fft.irfft(fft_from_top_n)

    # --- STAGE 5: Save and Play the Reconstructed Audio ---
    print(f"🔊 Saving and playing back the sound...")
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

    # --- STAGE 6: Export the Top N data to CSV (with rounding) ---
    print(f"📊 Exporting {NUM_TOP_FREQUENCIES} data points to CSV...")

    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)

    # Get the data only for our top N indices
    top_freqs = frequency_axis[top_n_indices]
    top_mags = full_magnitudes[top_n_indices]
    top_complex = fft_full[top_n_indices]

    fft_dataframe_top_n = pd.DataFrame({
        "Frequency (Hz)": top_freqs,
        "Magnitude": top_mags,
        "Real Part": top_complex.real,
        "Imaginary Part": top_complex.imag
    })

    # Sort by frequency for easier reading and round the data
    fft_dataframe_top_n = fft_dataframe_top_n.sort_values(by="Frequency (Hz)").round(2)

    fft_dataframe_top_n.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"   -> Saved {len(fft_dataframe_top_n)} data points to '{OUTPUT_CSV_FILENAME}'")

    # --- STAGE 7: Plotting ---
    print("📈 Generating final plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Signal Reconstruction from only the Top {NUM_TOP_FREQUENCIES} Strongest Frequencies", fontsize=16)
    time_axis = np.linspace(0, DURATION, num=N)

    axs[0, 0].plot(frequency_axis, full_magnitudes, color='magenta', lw=1)
    axs[0, 0].set_title("1. Full Frequency Spectrum")
    axs[0, 0].set_xlim(0, 8000)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    axs[0, 1].plot(frequency_axis, np.abs(fft_from_top_n), color='yellow', lw=1)
    axs[0, 1].set_title(f"2. Reduced Spectrum (Top {NUM_TOP_FREQUENCIES} Points Kept)")
    axs[0, 1].set_xlim(0, 8000)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    axs[1, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[1, 0].set_title("3. Original Recorded Signal")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.3, label='Original')
    axs[1, 1].plot(time_axis, audio_reconstructed, color='red', lw=1,
                   label=f'Reconstructed from Top {NUM_TOP_FREQUENCIES} points')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    top_n_reconstruction_cycle()