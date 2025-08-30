import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DURATION = 5  # seconds
SAMPLE_RATE = 44100  # Hertz
CHANNELS = 1  # Mono

# This factor controls how much we sample the FFT.
# A value of 10 means we keep 1 out of every 10 frequency components.
# Try changing this value (e.g., 2, 5, 20) to see how it affects the result!
SAMPLING_FACTOR = 10

# Force Matplotlib to use a reliable backend to avoid environment-specific bugs
import matplotlib

matplotlib.use('TkAgg')


def record_sample_reconstruct():
    """
    Records audio, samples its Fourier Transform, and reconstructs the signal
    to show the effect of information loss.
    """
    # --- STAGE 1: Record the Original Signal ---
    print(f"🎙️  Recording for {DURATION} seconds...")
    audio_original = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
    sd.wait()
    audio_original = audio_original.flatten()
    print("✅ Recording complete.")

    # --- STAGE 2: Calculate the Full Fourier Transform ---
    print("🔬 Analyzing full frequency spectrum...")
    N = len(audio_original)
    fft_full = np.fft.rfft(audio_original)

    # --- STAGE 3: Sample the Fourier Transform ---
    print(f"✂️  Sampling the spectrum (keeping 1 in every {SAMPLING_FACTOR} points)...")

    # Create a copy to modify. We need the original later.
    fft_sampled = fft_full.copy()

    # Create a mask to zero out the values we want to discard.
    # The mask is True for indices we want to set to 0.
    mask = np.arange(len(fft_sampled)) % SAMPLING_FACTOR != 0
    fft_sampled[mask] = 0

    # --- STAGE 4: Reconstruct the Signal from the Sampled FFT ---
    print("🏗️  Reconstructing signal from sampled spectrum...")
    audio_reconstructed = np.fft.irfft(fft_sampled)

    # --- Plotting All Stages ---
    print("📊 Generating plots...")

    # Create time and frequency axes for plotting
    time_axis = np.linspace(0, DURATION, num=N)
    frequency_axis = np.fft.rfftfreq(N, 1 / SAMPLE_RATE)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Signal Reconstruction from a Sampled Fourier Transform", fontsize=16)

    # Plot 1: Original Sound Wave
    axs[0, 0].plot(time_axis, audio_original, color='cyan', lw=1)
    axs[0, 0].set_title("1. Original Recorded Signal")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Full Frequency Spectrum
    axs[0, 1].plot(frequency_axis, np.abs(fft_full), color='magenta', lw=1)
    axs[0, 1].set_title("2. Full Frequency Spectrum (FFT)")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Magnitude")
    axs[0, 1].set_xlim(0, 5000)
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Sampled Frequency Spectrum
    axs[1, 0].plot(frequency_axis, np.abs(fft_sampled), color='yellow', lw=1)
    axs[1, 0].set_title(f"3. Sampled Spectrum (1 in {SAMPLING_FACTOR} kept)")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].set_xlim(0, 5000)
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Original vs. Reconstructed Signal
    axs[1, 1].plot(time_axis, audio_original, color='cyan', lw=2, alpha=0.4, label='Original')
    axs[1, 1].plot(time_axis, audio_reconstructed, color='red', lw=1, label='Reconstructed')
    axs[1, 1].set_title("4. Reconstructed Signal vs. Original")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Amplitude")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    record_sample_reconstruct()