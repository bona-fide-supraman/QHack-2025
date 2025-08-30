import sounddevice as sd
import numpy as np
import pandas as pd
import cv2
import queue
import time

# --- Configuration ---
CONFIG = {
    "SAMPLE_RATE": 4000,  # High sample rate for good frequency resolution
    "CHANNELS": 1,  # Mono audio
    "INTERVAL_MS": 200,  # Process audio every 200 milliseconds
    "NUM_LEDS": 25,  # Total LEDs (must be a perfect square, e.g., 25 for 5x5)
    "MIN_FREQ_HZ": 50,  # The lowest frequency band to analyze
    "MAX_FREQ_HZ": 8000,  # The highest frequency band to analyze
    "THRESHOLD_FACTOR": 0.4,  # Sensitivity. Lower is more sensitive (0.1-0.9)
    "OUTPUT_CSV_FILENAME": "led_spectrum_log.csv"
}

# --- Global Variables ---
audio_queue = queue.Queue()  # For safe thread communication
led_history = []  # To store the log for the CSV


def create_log_freq_bands(fft_size, sample_rate):
    """
    Creates 25 frequency bands on a logarithmic scale.
    Returns a list of (start_index, end_index) for each band in the FFT array.
    """
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # Create logarithmically spaced frequency points
    log_freq_points = np.logspace(
        np.log10(CONFIG["MIN_FREQ_HZ"]),
        np.log10(CONFIG["MAX_FREQ_HZ"]),
        CONFIG["NUM_LEDS"] + 1
    )

    bands = []
    for i in range(CONFIG["NUM_LEDS"]):
        start_freq = log_freq_points[i]
        end_freq = log_freq_points[i + 1]

        # Find the FFT indices that correspond to these frequencies
        start_index = np.searchsorted(freqs, start_freq, side='left')
        end_index = np.searchsorted(freqs, end_freq, side='right')
        bands.append((start_index, end_index))

    print("--- Frequency Bands (Hz) ---")
    for i, (start_idx, end_idx) in enumerate(bands):
        start_f = freqs[start_idx]
        end_f = freqs[end_idx - 1] if end_idx > start_idx else freqs[start_idx]
        print(f"LED {i:02d}: {start_f:7.1f} - {end_f:7.1f} Hz")
    print("----------------------------")
    return bands


def draw_led_grid(canvas, states):
    """Draws the 5x5 LED grid on an OpenCV canvas."""
    grid_size = int(np.sqrt(CONFIG["NUM_LEDS"]))
    height, width, _ = canvas.shape
    padding = width // 20
    cell_size = (width - 2 * padding) // grid_size

    for i, state in enumerate(states):
        row = i // grid_size
        col = i % grid_size
        x = padding + col * cell_size
        y = padding + row * cell_size

        color = (0, 255, 255) if state == 1 else (50, 50, 50)  # Yellow for ON, Gray for OFF
        cv2.circle(canvas, (x + cell_size // 2, y + cell_size // 2), cell_size // 3, color, -1)


def audio_callback(indata, frames, time, status):
    """This is called by sounddevice for each new audio chunk."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def main():
    """Main function to run the real-time analyzer."""
    # --- Initialization ---
    samples_per_interval = int(CONFIG["SAMPLE_RATE"] * (CONFIG["INTERVAL_MS"] / 1000.0))
    FFT_SIZE = int(2 ** np.ceil(np.log2(samples_per_interval)))  # Use next power of 2 for FFT

    freq_bands = create_log_freq_bands(FFT_SIZE, CONFIG["SAMPLE_RATE"])

    # This buffer will hold the most recent audio data
    audio_buffer = np.zeros(FFT_SIZE, dtype=np.float32)

    # Hanning window reduces FFT artifacts
    hanning_window = np.hanning(FFT_SIZE)

    try:
        # --- Start Audio Stream ---
        stream = sd.InputStream(
            samplerate=CONFIG["SAMPLE_RATE"],
            channels=CONFIG["CHANNELS"],
            callback=audio_callback
        )
        stream.start()
        print(f"\n🎙️  Listening... Press 'q' in the LED window to stop and save CSV.")

        while True:
            # --- Get Audio Data ---
            while not audio_queue.empty():
                chunk = audio_queue.get().flatten()
                # Roll the buffer and add the new chunk at the end
                audio_buffer = np.roll(audio_buffer, -len(chunk))
                audio_buffer[-len(chunk):] = chunk

            # --- Perform Analysis ---
            windowed_data = audio_buffer * hanning_window
            fft_data = np.fft.rfft(windowed_data)
            magnitudes = np.abs(fft_data)

            # Dynamic threshold based on the loudest frequency in the chunk
            dynamic_threshold = np.max(magnitudes) * CONFIG["THRESHOLD_FACTOR"]

            led_states = np.zeros(CONFIG["NUM_LEDS"], dtype=int)
            for i, (start_idx, end_idx) in enumerate(freq_bands):
                if start_idx < end_idx:
                    peak_magnitude_in_band = np.max(magnitudes[start_idx:end_idx])
                    if peak_magnitude_in_band > dynamic_threshold:
                        led_states[i] = 1

            # --- Log and Visualize ---
            timestamp = time.time()
            led_history.append([timestamp] + led_states.tolist())

            canvas = np.zeros((400, 400, 3), dtype=np.uint8)
            draw_led_grid(canvas, led_states)
            cv2.imshow("Real-time LED Spectrum Analyzer", canvas)

            # --- Control Loop ---
            if cv2.waitKey(CONFIG["INTERVAL_MS"]) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- Cleanup and Save ---
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        cv2.destroyAllWindows()
        print("\n🛑  Stream stopped.")

        if led_history:
            print(f"💾  Saving {len(led_history)} data points to '{CONFIG['OUTPUT_CSV_FILENAME']}'...")
            columns = ['Timestamp (s)'] + [f'LED_{i}' for i in range(CONFIG['NUM_LEDS'])]
            df = pd.DataFrame(led_history, columns=columns)
            df.to_csv(CONFIG['OUTPUT_CSV_FILENAME'], index=False, float_format='%.3f')
            print("✅  Save complete.")


if __name__ == '__main__':
    main()