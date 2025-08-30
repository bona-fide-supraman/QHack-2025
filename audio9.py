import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 4000  # Using a high sample rate for better quality
CHANNELS = 1  # Mono

# --- STFT Configuration ---
CHUNK_DURATION_MS = 200  # Process the audio in 200ms chunks
NUM_TOP_FREQUENCIES = 49  # Keep the top 25 frequencies for each chunk

# --- File Names ---
OUTPUT_WAV_FILENAME = f"stft_reconstructed_top_{NUM_TOP_FREQUENCIES}.wav"

import matplotlib

matplotlib.use('TkAgg')


def stft_reconstruction_cycle():
    """
    Performs a Short-Time Fourier Transform (STFT) analysis, keeping only the
    top N frequencies for each time slice, and reconstructs the audio.
    """
    # --- STAGE 1: Record the Original Signal ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    # --- STAGE 2: Process Audio in Chunks (STFT) ---
    print(f"🔬 Analyzing audio in {CHUNK_DURATION_MS}ms chunks...")

    # Calculate chunk size in samples
    chunk_size = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))
    num_chunks = len(audio_original) // chunk_size

    reconstructed_chunks = []
    spectrogram_data = []  # To store data for the spectrogram plot

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = audio_original[start_index:end_index]

        # --- Analysis within each chunk ---
        fft_chunk = np.fft.rfft(chunk)
        magnitudes = np.abs(fft_chunk)

        # Find the top N frequencies for this specific chunk
        sorted_indices = np.argsort(magnitudes)
        top_n_indices = sorted_indices[-NUM_TOP_FREQUENCIES:]

        # Create a sparse FFT for this chunk
        fft_chunk_reduced = np.zeros_like(fft_chunk)
        fft_chunk_reduced[top_n_indices] = fft_chunk[top_n_indices]

        # Store the reduced spectrum for our spectrogram plot
        spectrogram_data.append(np.abs(fft_chunk_reduced))

        # Reconstruct this specific chunk
        reconstructed_chunk = np.fft.irfft(fft_chunk_reduced)
        reconstructed_chunks.append(reconstructed_chunk)

    # --- STAGE 3: Stitch Reconstructed Chunks Together ---
    print("🏗️  Stitching reconstructed audio chunks...")
    audio_reconstructed = np.concatenate(reconstructed_chunks)

    # --- STAGE 4: Save and Play the Reconstructed Audio ---
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
    fig.suptitle(f"STFT Reconstruction (Top {NUM_TOP_FREQUENCIES} Frequencies per {CHUNK_DURATION_MS}ms Chunk)",
                 fontsize=16)

    N_full = len(audio_original)
    time_axis = np.linspace(0, DURATION, num=N_full)

    # Plot 1: Full Frequency Spectrum (of the entire 5s clip)
    fft_full = np.fft.rfft(audio_original)
    freq_axis_full = np.fft.rfftfreq(N_full, 1 / SAMPLE_RATE)
    axs[0, 0].plot(freq_axis_full, np.abs(fft_full), color='magenta', lw=1)
    axs[0, 0].set_title("1. Full Spectrum (of entire 5s recording)")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_xlim(0, 8000)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: The Reduced Spectrogram
    # Transpose the data so time is on the x-axis
    spectrogram = np.array(spectrogram_data).T
    freq_axis_chunk = np.fft.rfftfreq(chunk_size, 1 / SAMPLE_RATE)
    time_bins = np.arange(num_chunks) * (CHUNK_DURATION_MS / 1000.0)

    # Use pcolormesh for the spectrogram
    axs[0, 1].pcolormesh(time_bins, freq_axis_chunk, spectrogram, shading='gouraud', cmap='inferno')
    axs[0, 1].set_title("2. Reduced Spectrogram (Top 25 Freqs over Time)")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylim(0, 8000)  # Set y-limit to match other plots

    # Plot 3: Original Recorded Signal
    axs[1, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[1, 0].set_title("3. Original Recorded Signal")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Reconstructed Signal vs. Original
    # We need a new time axis for the reconstructed signal as it might be slightly shorter
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
    stft_reconstruction_cycle()