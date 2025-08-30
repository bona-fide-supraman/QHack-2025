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

# --- NEW CONFIGURATION: Further reduced for an even smaller file ---
# We are now asking for only the 100 most prominent peaks across the spectrum.
NUM_CSV_DATA_POINTS = 100

# --- File Names ---
OUTPUT_WAV_FILENAME = "reconstructed_audio.wav"
# Updated filename to reflect the change
OUTPUT_CSV_FILENAME = "sampled_fft_data_highly_reduced.csv"

import matplotlib

matplotlib.use('TkAgg')


def full_audio_analysis_cycle_with_extreme_reduction():
    """
    Records audio, performs smart FFT sampling, reconstructs the signal,
    plays the audio, and saves a HIGHLY downsampled version of the FFT data.
    """
    # --- Stages 1, 2, 3, and 4 are identical to the previous script ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    print("🔬 Analyzing and sampling the frequency spectrum...")
    N = len(audio_original)
    fft_full = np.fft.rfft(audio_original)
    magnitudes = np.abs(fft_full)

    threshold = np.percentile(magnitudes, 100 * (1 - PERCENTAGE_TO_KEEP))
    mask = magnitudes < threshold
    fft_sampled = fft_full.copy()
    fft_sampled[mask] = 0

    print("🏗️  Reconstructing signal from the sampled spectrum...")
    audio_reconstructed = np.fft.irfft(fft_sampled)

    print(f"🔊 Saving and playing back the reconstructed sound...")
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

    # --- STAGE 5 (REVISED): Downsample FFT Data and Export to CSV ---
    print(f"📊 Downsampling FFT data to {NUM_CSV_DATA_POINTS} points and exporting...")

    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)
    sampled_magnitudes = np.abs(fft_sampled)

    # --- Smart Binning Logic (works the same, just with larger bins) ---
    num_full_points = len(frequency_axis)
    bin_size = num_full_points // NUM_CSV_DATA_POINTS

    binned_freqs, binned_mags, binned_reals, binned_imags = [], [], [], []

    for i in range(NUM_CSV_DATA_POINTS):
        start_index = i * bin_size
        end_index = start_index + bin_size
        mag_slice = sampled_magnitudes[start_index:end_index]
        if len(mag_slice) > 0:
            max_index_in_slice = np.argmax(mag_slice)
            peak_index = start_index + max_index_in_slice

            binned_freqs.append(frequency_axis[peak_index])
            binned_mags.append(sampled_magnitudes[peak_index])
            binned_reals.append(fft_sampled[peak_index].real)
            binned_imags.append(fft_sampled[peak_index].imag)

    fft_dataframe_reduced = pd.DataFrame({
        "Frequency (Hz)": binned_freqs,
        "Magnitude": binned_mags,
        "Real Part": binned_reals,
        "Imaginary Part": binned_imags
    })

    fft_dataframe_reduced.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"   -> Saved {len(fft_dataframe_reduced)} data points to '{OUTPUT_CSV_FILENAME}'")

    # --- STAGE 6: Plotting (unchanged) ---
    print("📈 Generating final plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Signal Reconstruction (High-Res) and Data Export (Highly Downsampled)", fontsize=16)
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
    full_audio_analysis_cycle_with_extreme_reduction()