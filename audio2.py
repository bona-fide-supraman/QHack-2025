import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hertz
CHANNELS = 1  # Mono

# This factor now controls the PERCENTAGE of FFT components to KEEP.
# 0.1 means we keep the strongest 10%.
# Try 0.5 (50%) or 0.01 (1%) to see the difference!
PERCENTAGE_TO_KEEP = 0.1

import matplotlib

matplotlib.use('TkAgg')


def smart_reconstruction():
    """
    Records audio, keeps only the strongest frequency components, and reconstructs
    the signal to show a more robust form of compression.
    """
    # --- STAGE 1: Record the Original Signal ---
    print(f"🎙️  Recording for {DURATION} seconds... (Try a clap or a snap!)")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    # --- STAGE 2: Calculate the Full Fourier Transform ---
    print("🔬 Analyzing full frequency spectrum...")
    N = len(audio_original)
    fft_full = np.fft.rfft(audio_original)
    magnitudes = np.abs(fft_full)

    # --- STAGE 3: "Smarter" Sampling of the Fourier Transform ---
    print(f"✂️  Keeping the strongest {PERCENTAGE_TO_KEEP * 100:.0f}% of frequency components...")

    # Find the threshold magnitude. Any component with a magnitude below this
    # value will be discarded. We use np.percentile to find this value.
    # For example, percentile(..., 10) finds the value that 10% of the data is below.
    # So to keep the top 10%, we find the 90th percentile.
    threshold = np.percentile(magnitudes, 100 * (1 - PERCENTAGE_TO_KEEP))

    # Create a mask to zero out the weak components.
    # The mask is True for magnitudes that are BELOW the threshold.
    mask = magnitudes < threshold

    fft_sampled = fft_full.copy()
    fft_sampled[mask] = 0

    # --- STAGE 4: Reconstruct the Signal from the Sampled FFT ---
    print("🏗️  Reconstructing signal...")
    audio_reconstructed = np.fft.irfft(fft_sampled)

    # --- Plotting All Stages ---
    print("📊 Generating plots...")
    time_axis = np.linspace(0, DURATION, num=N)
    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Signal Reconstruction using 'Strongest Frequency' Sampling", fontsize=16)

    # Plot 1 & 2 are the same as before
    axs[0, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[0, 0].set_title("1. Original Recorded Signal")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    axs[0, 1].plot(frequency_axis, magnitudes, color='magenta', lw=1)
    axs[0, 1].set_title("2. Full Frequency Spectrum (FFT)")
    axs[0, 1].set_xlim(0, 8000)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Sampled Frequency Spectrum (will look more interesting now)
    axs[1, 0].plot(frequency_axis, np.abs(fft_sampled), color='yellow', lw=1)
    axs[1, 0].set_title(f"3. Sampled Spectrum (Strongest {PERCENTAGE_TO_KEEP * 100:.0f}% Kept)")
    axs[1, 0].set_xlim(0, 8000)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Original vs. Reconstructed Signal
    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.4, label='Original')
    axs[1, 1].plot(time_axis, audio_reconstructed, color='red', lw=1, label='Reconstructed')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    smart_reconstruction()