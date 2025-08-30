import argparse
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, stft, istft
from scipy.io.wavfile import write
from datetime import datetime


def record_audio(duration_ms, samplerate=44100):
    duration_s = duration_ms / 1000.0
    print(f"Recording {duration_s} seconds of audio...")
    audio = sd.rec(int(samplerate * duration_s), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), samplerate


def save_wav(audio, samplerate, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}{suffix}.wav"
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    write(filename, samplerate, audio_int16)
    print(f"Audio saved to {filename}")
    return filename


def plot_waterfall(audio, samplerate, num_columns, log_scale, fmin, fmax):
    f, t, Sxx = spectrogram(audio, samplerate, nperseg=1024, noverlap=512)
    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]

    num_time_bins = len(t)
    bins_per_col = num_time_bins // num_columns

    partitioned_spectrogram = []
    partitioned_times = []
    for i in range(num_columns):
        start = i * bins_per_col
        end = (i + 1) * bins_per_col if i < num_columns - 1 else num_time_bins
        partitioned_spectrogram.append(np.mean(Sxx[:, start:end], axis=1))
        partitioned_times.append(np.mean(t[start:end]))

    partitioned_spectrogram = np.array(partitioned_spectrogram).T

    if log_scale:
        partitioned_spectrogram = 10 * np.log10(partitioned_spectrogram + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.imshow(partitioned_spectrogram,
               aspect='auto',
               origin='lower',
               extent=[partitioned_times[0], partitioned_times[-1], f[0], f[-1]])

    plt.colorbar(label='Power (dB)' if log_scale else 'Power (Linear)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Waterfall Spectrogram (Partitioned)')
    plt.show()


def remove_phase_and_reconstruct(audio, samplerate):
    # STFT
    f, t, Zxx = stft(audio, fs=samplerate, nperseg=1024, noverlap=512)
    # Remove phase (keep magnitude only)
    magnitude = np.abs(Zxx)
    Zxx_no_phase = magnitude  # real and nonnegative
    # Inverse STFT
    _, reconstructed = istft(Zxx_no_phase, fs=samplerate, nperseg=1024, noverlap=512)
    return reconstructed


def main():
    parser = argparse.ArgumentParser(description="Waterfall Spectrogram from Microphone Recording")
    parser.add_argument("--duration-ms", type=int, default=1000, help="Recording duration in milliseconds")
    parser.add_argument("--num-columns", type=int, default=10, help="Number of spectrogram columns")
    parser.add_argument("--log", action="store_true", help="Plot in decibel scale")
    parser.add_argument("--min", type=float, default=0.0, help="Minimum frequency to plot (Hz)")
    parser.add_argument("--max", type=float, default=20000.0, help="Maximum frequency to plot (Hz)")
    args = parser.parse_args()

    audio, samplerate = record_audio(args.duration_ms)
    save_wav(audio, samplerate)  # original

    reconstructed = remove_phase_and_reconstruct(audio, samplerate)
    save_wav(reconstructed, samplerate, suffix="_no_phase")  # reconstructed

    plot_waterfall(audio, samplerate, args.num_columns, args.log, args.min, args.max)


if __name__ == "__main__":
    main()
