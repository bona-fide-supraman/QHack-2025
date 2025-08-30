import sys
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram, stft, istft
from scipy.io.wavfile import write
from datetime import datetime

# PyQt5 and Matplotlib integration imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QFormLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# --- Your original audio processing functions (with minor adjustments) ---

def record_audio(duration_ms, samplerate=44100):
    duration_s = duration_ms / 1000.0
    print(f"Recording {duration_s} seconds of audio...")
    audio = sd.rec(int(samplerate * duration_s), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), samplerate


def save_wav(audio, samplerate, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}{suffix}.wav"
    # Ensure audio is not empty before trying to normalize
    if audio.size > 0 and np.max(np.abs(audio)) > 0:
        audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    else:
        audio_int16 = np.int16(audio)
    write(filename, samplerate, audio_int16)
    print(f"Audio saved to {filename}")
    return filename


def remove_phase_and_reconstruct(audio, samplerate):
    f, t, Zxx = stft(audio, fs=samplerate, nperseg=1024, noverlap=512)
    magnitude = np.abs(Zxx)
    Zxx_no_phase = magnitude
    _, reconstructed = istft(Zxx_no_phase, fs=samplerate, nperseg=1024, noverlap=512)
    return reconstructed


# --- PyQt Application Class ---

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waterfall Spectrogram Analyzer")
        self.setGeometry(100, 100, 800, 700)  # x, y, width, height

        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Controls Layout (Left Side) ---
        self.controls_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.duration_input = QSpinBox()
        self.duration_input.setRange(100, 60000)
        self.duration_input.setValue(1000)
        self.duration_input.setSuffix(" ms")
        form_layout.addRow("Recording Duration:", self.duration_input)

        self.columns_input = QSpinBox()
        self.columns_input.setRange(1, 100)
        self.columns_input.setValue(10)
        form_layout.addRow("Spectrogram Columns:", self.columns_input)

        self.fmin_input = QDoubleSpinBox()
        self.fmin_input.setRange(0, 22000)
        self.fmin_input.setValue(0.0)
        self.fmin_input.setSuffix(" Hz")
        form_layout.addRow("Min Frequency:", self.fmin_input)

        self.fmax_input = QDoubleSpinBox()
        self.fmax_input.setRange(0, 22000)
        self.fmax_input.setValue(20000.0)
        self.fmax_input.setSuffix(" Hz")
        form_layout.addRow("Max Frequency:", self.fmax_input)

        self.log_checkbox = QCheckBox("Plot in Decibel Scale")
        form_layout.addRow(self.log_checkbox)

        self.run_button = QPushButton("Record and Analyze")
        self.run_button.clicked.connect(self.run_analysis)

        self.controls_layout.addLayout(form_layout)
        self.controls_layout.addWidget(self.run_button)
        self.controls_layout.addStretch()  # Pushes controls to the top

        # --- Plot Layout (Right Side) ---
        self.plot_layout = QVBoxLayout()
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # The plotting area
        self.plot_layout.addWidget(self.canvas)

        # --- Combine Layouts ---
        self.main_layout.addLayout(self.controls_layout, 1)  # 1 part width
        self.main_layout.addLayout(self.plot_layout, 3)  # 3 parts width

        self.show()

    def run_analysis(self):
        """Reads values from the UI, runs the audio processing, and updates the plot."""
        # 1. Get parameters from the GUI
        duration = self.duration_input.value()
        num_cols = self.columns_input.value()
        log_scale = self.log_checkbox.isChecked()
        fmin = self.fmin_input.value()
        fmax = self.fmax_input.value()

        # 2. Run the core logic
        audio, samplerate = record_audio(duration)
        save_wav(audio, samplerate)

        reconstructed = remove_phase_and_reconstruct(audio, samplerate)
        save_wav(reconstructed, samplerate, suffix="_no_phase")

        # 3. Update the plot
        self.plot_waterfall(audio, samplerate, num_cols, log_scale, fmin, fmax)

    def plot_waterfall(self, audio, samplerate, num_columns, log_scale, fmin, fmax):
        """Modified plot function to draw on the GUI's canvas."""
        self.ax.clear()  # Clear the previous plot

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

        im = self.ax.imshow(partitioned_spectrogram,
                            aspect='auto',
                            origin='lower',
                            extent=[partitioned_times[0], partitioned_times[-1], f[0], f[-1]])

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.set_title('Waterfall Spectrogram (Partitioned)')

        # Add a colorbar
        self.figure.colorbar(im, ax=self.ax, label='Power (dB)' if log_scale else 'Power (Linear)')

        self.canvas.draw()  # Redraw the canvas to show the new plot


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())