Time Series Anomaly Detection Using Deep Learning

https://se.mathworks.com/help/deeplearning/ug/time-series-anomaly-detection-using-deep-learning.html

translated from Pytorch to TensorFlow, by Claude

1. Model Architecture

Replaced PyTorch nn.Module with TensorFlow keras.Model
Used layers.Conv1D and layers.Conv1DTranspose instead of PyTorch equivalents
TensorFlow handles the channel-last format natively (batch, time, channels)

2. Data Loading

Replaced PyTorch Dataset and DataLoader with custom WaveformSequence class inheriting from keras.utils.Sequence
Handles variable-length sequences with padding in the same way
No need for custom collate functions - handled within the sequence class

3. Training Loop

Replaced manual training loop with model.fit()
Used keras.callbacks.Callback for loss tracking
TensorFlow automatically handles device placement (CPU/GPU)

4. Key API Mappings

torch.FloatTensor → np.array(..., dtype=np.float32)
nn.MSELoss() → 'mse' loss function
optim.Adam → keras.optimizers.Adam
model.train()/eval() → handled automatically by TensorFlow
torch.no_grad() → not needed, use model.predict()
