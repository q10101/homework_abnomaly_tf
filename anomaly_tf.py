import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import pickle
import scipy.io
import os
import time
from datetime import datetime

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


class WaveformSequence(keras.utils.Sequence):
    """Custom sequence for waveform data with variable length support"""

    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indices]

        # Find max length in batch
        max_len = max([seq.shape[0] for seq in batch_data])

        # Pad sequences
        padded_inputs = []
        padded_targets = []

        for seq in batch_data:
            seq_len = seq.shape[0]
            if seq_len < max_len:
                pad_size = max_len - seq_len
                pad = np.zeros((pad_size, seq.shape[1]), dtype=np.float32)
                seq = np.concatenate([seq, pad], axis=0)

            padded_inputs.append(seq)
            padded_targets.append(seq)  # Autoencoder: input = target

        return np.array(padded_inputs), np.array(padded_targets)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class Conv1DAutoencoder(keras.Model):
    """1D Convolutional Autoencoder for time series anomaly detection"""

    def __init__(self, num_channels, filter_size=7, num_filters=16, dropout_prob=0.2, num_downsamples=2):
        super(Conv1DAutoencoder, self).__init__()

        self.num_channels = num_channels
        self.num_downsamples = num_downsamples

        # Encoder layers
        self.encoder_layers = []
        in_channels = num_channels

        for i in range(num_downsamples):
            out_channels = (num_downsamples - i) * num_filters
            self.encoder_layers.extend([
                layers.Conv1D(out_channels, filter_size, strides=2, padding='same', activation='relu'),
                layers.Dropout(dropout_prob)
            ])
            in_channels = out_channels

        # Decoder layers
        self.decoder_layers = []

        for i in range(num_downsamples):
            out_channels = (i + 1) * num_filters
            self.decoder_layers.extend([
                layers.Conv1DTranspose(out_channels, filter_size, strides=2, padding='same', activation='relu'),
                layers.Dropout(dropout_prob)
            ])
            in_channels = out_channels

        # Final reconstruction layer
        self.decoder_layers.append(
            layers.Conv1DTranspose(num_channels, filter_size, strides=1, padding='same')
        )

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, sequence_length, num_channels)
        x = inputs

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x, training=training)

        return x


def load_data():
    """Main function to prepare all data and set global variables."""
    # Use synthetic data if WaveformData.mat is not found, otherwise try to load it
    mat_file_path = 'WaveformData.mat'
    if os.path.exists(mat_file_path):
        data = read_waveform_data_from_mat(mat_file_path)
        if not data:
            print(f"Could not load data from {mat_file_path}, generating synthetic data instead.")
            data = generate_synthetic_data(num_observations=1000, num_channels=3)
    else:
        print(f"'{mat_file_path}' not found, generating synthetic data.")
        data = generate_synthetic_data(num_observations=1000, num_channels=3)

    if not data:
        raise RuntimeError("No data available for processing. Exiting.")

    num_channels = data[0].shape[1]
    print(f"Loaded {len(data)} observations with {num_channels} channels.")

    # Visualize the first few sequences
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(min(4, len(data))):
        ax = axes[i]
        for channel in range(num_channels):
            ax.plot(data[i][:, channel], label=f'Channel {channel + 1}')
        ax.set_xlabel("Time Step")
        ax.set_title(f"Sequence {i + 1}")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.suptitle("Example Waveform Sequences", y=1.02)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/1-example-waveform.png")
    plt.show(block=False)
    plt.pause(0.1)

    return data


def prepare_data(data, num_downsamples=2):
    """Prepare data by cropping to make divisible by 2^num_downsamples"""
    processed_data = []
    sequence_lengths = []

    for sequence in data:
        seq_len = len(sequence)
        cropping = seq_len % (2 ** num_downsamples)
        if cropping > 0:
            sequence = sequence[:-cropping]
        processed_data.append(sequence)
        sequence_lengths.append(len(sequence))

    return processed_data, sequence_lengths


def generate_synthetic_data(num_observations=1000, min_timesteps=50, max_timesteps=150, num_channels=3):
    """Generates synthetic waveform data for demonstration."""
    data = []
    for _ in range(num_observations):
        num_time_steps = np.random.randint(min_timesteps, max_timesteps + 1)
        time = np.linspace(0, 2 * np.pi, num_time_steps)
        sequence = np.zeros((num_time_steps, num_channels))
        for i in range(num_channels):
            amplitude = np.random.rand() * 2 + 1
            frequency = np.random.rand() * 2 + 0.5
            phase = np.random.rand() * 2 * np.pi
            noise = np.random.randn(num_time_steps) * 0.1
            sequence[:, i] = amplitude * np.sin(frequency * time + phase) + noise
        data.append(sequence.astype(np.float32))
    return data


def read_waveform_data_from_mat(file_path='WaveformData.mat'):
    """Reads waveform data from a .mat file."""
    try:
        mat_contents = scipy.io.loadmat(file_path)
        if 'data' in mat_contents:
            loaded_data = mat_contents['data']
            if isinstance(loaded_data, np.ndarray) and loaded_data.dtype == object:
                if loaded_data.ndim > 1:
                    loaded_data = loaded_data.flatten()
                return [seq.astype(np.float32) for seq in loaded_data if isinstance(seq, np.ndarray)]
            else:
                print(f"Warning: Data found under 'data' key is not in the expected format (NumPy object array).")
                if isinstance(loaded_data, np.ndarray):
                    return [loaded_data.astype(np.float32)]
                return []
        else:
            print(f"Error: 'data' key not found in {file_path}. Please check the .mat file structure.")
            return []
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the .mat file: {e}")
        return []


def plot_training_data(data, num_channels):
    """Plot first 4 observations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(min(4, len(data))):
        for ch in range(num_channels):
            axes[i].plot(data[i][:, ch], label=f'Channel {ch + 1}')
        axes[i].set_title(f'Observation {i + 1}')
        axes[i].set_xlabel('Time Step')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


class LossHistory(keras.callbacks.Callback):
    """Custom callback to track training history"""

    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def train_autoencoder(model, train_sequence, val_sequence, num_epochs=120, learning_rate=0.001):
    """Train the autoencoder"""

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Callbacks
    loss_history = LossHistory()

    print("\n--- Training AutoEncoder Neural Network ---")
    train_start_time = time.time()
    train_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at: {train_start_datetime}")

    # Train model
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=num_epochs,
        callbacks=[loss_history],
        verbose=1
    )

    train_end_time = time.time()
    training_elapsed_time = train_end_time - train_start_time
    print(f"Training finished in: {training_elapsed_time:.2f} seconds")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Training Loss per Epoch\nStart: {train_start_datetime}, Elapsed: {training_elapsed_time:.2f}s")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)

    return history.history['loss'], history.history['val_loss']


def calculate_rmse(model, data_sequence):
    """Calculate RMSE for each sequence"""
    rmse_values = []

    for i in range(len(data_sequence)):
        batch_x, batch_y = data_sequence[i]
        predictions = model.predict(batch_x, verbose=0)

        # Calculate RMSE for each sequence in the batch
        for j in range(batch_x.shape[0]):
            seq_len = batch_y.shape[1]
            pred_len = min(predictions.shape[1], seq_len)

            mse = np.mean((predictions[j, :pred_len, :] - batch_y[j, :pred_len, :]) ** 2)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)

    return np.array(rmse_values)


def plot_hist(val, ss):
    plt.figure(figsize=(8, 5))
    plt.hist(val, bins=20, alpha=0.7)
    plt.xlabel(ss)
    plt.ylabel('Frequency')
    plt.title('Representative Samples')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)


def plot_hist2(val_hist, val_line, ss_hist, ss_line):
    plt.figure(figsize=(8, 5))
    plt.hist(val_hist, bins=20, alpha=0.7, label=ss_hist)
    plt.axvline(val_line, color='red', linestyle='--', label=ss_line)
    plt.xlabel('Root Mean Square Error (RMSE)')
    plt.ylabel('Frequency')
    plt.title('New Samples')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1)


def inject_anomalies(data, num_anomalous=20, patch_start=50, patch_end=60, scale_factor=4):
    """Inject anomalies into random sequences"""
    data_with_anomalies = [seq.copy() for seq in data]
    anomaly_indices = random.sample(range(len(data)), num_anomalous)

    for idx in anomaly_indices:
        sequence = data_with_anomalies[idx]
        if len(sequence) > patch_end:
            patch = sequence[patch_start:patch_end, :]
            sequence[patch_start:patch_end, :] = scale_factor * np.abs(patch)

    return data_with_anomalies, anomaly_indices


def detect_anomalous_regions(original, reconstructed, rmse_baseline, window_size=7, threshold_factor=1.1):
    """Detect anomalous regions within a sequence"""
    # Calculate RMSE for each time step across all channels
    rmse_per_step = np.sqrt(np.mean((reconstructed - original) ** 2, axis=1))

    threshold = threshold_factor * rmse_baseline
    anomaly_mask = np.zeros(len(original), dtype=bool)

    # Apply sliding window
    for t in range(len(rmse_per_step) - window_size + 1):
        window_rmse = rmse_per_step[t:t + window_size]
        if np.all(window_rmse > threshold):
            anomaly_mask[t:t + window_size] = True

    return anomaly_mask


def plot_reconstruction(original, reconstructed, sequence_idx):
    """Plot original vs reconstructed"""
    num_channels = original.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]

    fig.suptitle(f'Sequence {sequence_idx}')

    for ch in range(num_channels):
        min_len = min(len(original), len(reconstructed))
        axes[ch].plot(original[:min_len, ch], 'b-', label='Original' if ch == 0 else '')
        axes[ch].plot(reconstructed[:min_len, ch], 'r--', label='Reconstructed' if ch == 0 else '')
        axes[ch].set_ylabel(f'Channel {ch + 1}')
        axes[ch].grid(True)

        if ch == 0:
            axes[ch].legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_anomaly_detection(original, reconstructed, anomaly_mask, sequence_idx):
    """Plot original vs reconstructed with anomalous regions highlighted"""
    num_channels = original.shape[1]

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels))
    if num_channels == 1:
        axes = [axes]

    fig.suptitle(f'Anomaly Detection - Sequence {sequence_idx}')

    for ch in range(num_channels):
        axes[ch].plot(original[:, ch], 'b-', label='Input')
        axes[ch].plot(reconstructed[:, ch], 'g--', label='Reconstructed' if ch == 0 else '')

        # Highlight anomalous regions
        anomalous_signal = np.full_like(original[:, ch], np.nan)
        anomalous_signal[anomaly_mask] = original[anomaly_mask, ch]
        axes[ch].plot(anomalous_signal, 'r-', linewidth=3, label='Anomalous' if ch == 0 else '')

        axes[ch].set_ylabel(f'Channel {ch + 1}')
        axes[ch].grid(True)

        if ch == 0:
            axes[ch].legend()

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    # Generate fake data or load existing data
    print("Loading data from mat file...")
    data = load_data()
    num_channels = data[0].shape[1]
    print(f"num_channels = {num_channels}")

    # Plot training data
    print("Plotting training data...")
    plot_training_data(data, num_channels)

    # Split data
    num_observations = len(data)
    split_idx = int(0.9 * num_observations)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Prepare data
    print("Preparing data...")
    train_data, train_seq_lengths = prepare_data(train_data)
    val_data, _ = prepare_data(val_data)

    # Create data sequences
    train_sequence = WaveformSequence(train_data, batch_size=8, shuffle=True)
    val_sequence = WaveformSequence(val_data, batch_size=8, shuffle=False)

    # Create model
    min_length = min(train_seq_lengths)
    model = Conv1DAutoencoder(num_channels)

    # Build model by calling it with sample data
    sample_input = np.random.random((1, min_length, num_channels)).astype(np.float32)
    _ = model(sample_input)

    print(f"Model architecture:")
    model.summary()

    # Train model
    print("Training autoencoder...")
    train_losses, val_losses = train_autoencoder(model, train_sequence, val_sequence)

    # Calculate baseline RMSE
    print("Calculating baseline RMSE...")
    val_rmse = calculate_rmse(model, val_sequence)

    plot_hist(val_rmse, 'Root Mean Square Error (RMSE)')

    rmse_baseline = np.max(val_rmse)
    print(f"RMSE Baseline: {rmse_baseline:.4f}")

    # Inject anomalies
    print("Injecting anomalies...")
    anomalous_data, anomaly_indices = inject_anomalies(val_data, num_anomalous=20)

    # Test on anomalous data
    anomalous_sequence = WaveformSequence(anomalous_data, batch_size=8, shuffle=False)
    anomalous_rmse = calculate_rmse(model, anomalous_sequence)

    plot_hist2(anomalous_rmse, rmse_baseline, 'data', 'RMSE-baseline')

    # Find top anomalous sequences
    top_indices = np.argsort(anomalous_rmse)[::-1]
    n_top = 10
    print(f"Top {n_top} anomalous sequence indices: {top_indices[:n_top]}")

    # Analyze most anomalous sequence
    random.seed(None)  # Uses system time (default behavior)
    most_anomalous_idx = top_indices[random.randint(0, n_top - 1)]
    original_seq = np.expand_dims(anomalous_data[most_anomalous_idx], axis=0).astype(np.float32)

    reconstructed_seq = model.predict(original_seq, verbose=0)

    original_seq = original_seq.squeeze(0)
    reconstructed_seq = reconstructed_seq.squeeze(0)

    # Detect anomalous regions
    print("Detecting anomalous regions...")
    min_len = min(len(original_seq), len(reconstructed_seq))
    anomaly_mask = detect_anomalous_regions(
        original_seq[:min_len],
        reconstructed_seq[:min_len],
        rmse_baseline
    )

    plot_reconstruction(original_seq[:min_len], reconstructed_seq[:min_len], most_anomalous_idx)
    plot_anomaly_detection(original_seq[:min_len], reconstructed_seq[:min_len], anomaly_mask, most_anomalous_idx)

    print("Done.")


if __name__ == "__main__":
    main()