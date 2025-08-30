import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 4000  # Using a high sample rate for good frequency resolution
CHANNELS = 1  # Mono

# --- STFT Configuration ---
CHUNK_DURATION_MS = 200  # Process the audio in 200ms chunks
NUM_TOP_FREQUENCIES = 49  # Keep the top N frequencies for each chunk

# --- NEW: Frequency Range Limiting ---
MIN_FREQ_HZ = 200.0
MAX_FREQ_HZ = 2000.0

# --- File Names ---
OUTPUT_WAV_FILENAME = f"stft_reconstructed_top_{NUM_TOP_FREQUENCIES}_filtered.wav"

import matplotlib

matplotlib.use('TkAgg')


def stft_reconstruction_cycle_filtered():
    """
    Performs a Short-Time Fourier Transform (STFT) analysis within a specific
    frequency range, keeping the top N frequencies, and reconstructs the audio.
    """
    # --- STAGE 1: Record the Original Signal ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    # --- STAGE 2: Process Audio in Chunks (STFT) ---
    print(f"🔬 Analyzing audio in {CHUNK_DURATION_MS}ms chunks between {MIN_FREQ_HZ} and {MAX_FREQ_HZ} Hz...")

    chunk_size = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))
    num_chunks = len(audio_original) // chunk_size

    reconstructed_chunks = []
    spectrogram_data = []

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = audio_original[start_index:end_index]

        # --- Analysis within each chunk ---
        fft_chunk = np.fft.rfft(chunk)

        # --- NEW: Apply Frequency Limiting ---
        # Create a frequency axis for this chunk
        freq_axis_chunk = np.fft.rfftfreq(chunk_size, 1 / SAMPLE_RATE)

        # Find the start and end indices for our frequency range
        freq_start_index = np.searchsorted(freq_axis_chunk, MIN_FREQ_HZ, side='left')
        freq_end_index = np.searchsorted(freq_axis_chunk, MAX_FREQ_HZ, side='right')

        # Isolate the magnitudes ONLY within our target range
        magnitudes_in_range = np.abs(fft_chunk[freq_start_index:freq_end_index])

        # Find the top N frequencies relative to this isolated range
        # We must handle the case where we ask for more frequencies than are available in the range
        num_peaks_to_find = min(NUM_TOP_FREQUENCIES, len(magnitudes_in_range))

        sorted_indices_in_range = np.argsort(magnitudes_in_range)
        top_n_indices_in_range = sorted_indices_in_range[-num_peaks_to_find:]

        # Convert these relative indices back to absolute indices of the full fft_chunk
        top_n_indices_absolute = top_n_indices_in_range + freq_start_index

        # Create a sparse FFT for this chunk
        fft_chunk_reduced = np.zeros_like(fft_chunk)
        # Place the peaks in their correct positions
        fft_chunk_reduced[top_n_indices_absolute] = fft_chunk[top_n_indices_absolute]

        spectrogram_data.append(np.abs(fft_chunk_reduced))

        # Reconstruct this specific chunk
        reconstructed_chunk = np.fft.irfft(fft_chunk_reduced)
        reconstructed_chunks.append(reconstructed_chunk)

    # --- STAGE 3 & 4 (Unchanged) ---
    print("🏗️  Stitching reconstructed audio chunks...")
    audio_reconstructed = np.concatenate(reconstructed_chunks)

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

    # --- STAGE 5: Plotting ---
    print("📈 Generating final plots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"STFT Reconstruction ({MIN_FREQ_HZ:.0f}-{MAX_FREQ_HZ:.0f} Hz, Top {NUM_TOP_FREQUENCIES} Freqs per Chunk)",
        fontsize=16)

    N_full = len(audio_original)
    time_axis = np.linspace(0, DURATION, num=N_full)

    fft_full = np.fft.rfft(audio_original)
    freq_axis_full = np.fft.rfftfreq(N_full, 1 / SAMPLE_RATE)
    axs[0, 0].plot(freq_axis_full, np.abs(fft_full), color='magenta', lw=1)
    axs[0, 0].set_title("1. Full Spectrum (of entire 5s recording)")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_xlim(0, SAMPLE_RATE / 2)
    # Add vertical lines to show the filtered range
    axs[0, 0].axvline(MIN_FREQ_HZ, color='lime', linestyle='--', alpha=0.7)
    axs[0, 0].axvline(MAX_FREQ_HZ, color='lime', linestyle='--', alpha=0.7)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    spectrogram = np.array(spectrogram_data).T
    time_bins = np.arange(num_chunks) * (CHUNK_DURATION_MS / 1000.0)

    axs[0, 1].pcolormesh(time_bins, freq_axis_chunk, spectrogram, shading='gouraud', cmap='inferno')
    axs[0, 1].set_title("2. Reduced Spectrogram (Processed Range)")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_xlabel("Time (s)")
    # Zoom the y-axis to our area of interest
    axs[0, 1].set_ylim(0, MAX_FREQ_HZ + 200)

    axs[1, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[1, 0].set_title("3. Original Recorded Signal")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    time_axis_recon = np.linspace(0, len(audio_reconstructed) / SAMPLE_RATE, num=len(audio_reconstructed))
    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.3, label='Original')
    axs[1, 1].plot(time_axis_recon, audio_reconstructed, color='red', lw=1, label='Reconstructed')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    stft_reconstruction_cycle_filtered()