# Unsupervised Representation Learning

This module trains a neural network to learn compact, fixed-length vector representations (embeddings) from variable-length acoustic feature sequences.

## 🎯 Objective

To compress the complex, temporal dynamics of a participant's speech (extracted by OpenSMILE) into a single latent vector that captures their "vocal style" or "interaction signature," without relying on labeled ground truth data.

## 🤖 Model Architecture

We utilize a 1D Convolutional Autoencoder (Conv1D_AE) designed specifically for multivariate time-series data.

### 1. The Encoder (Compression)

#### Input

A sequence of acoustic features of shape (Batch, 88, Time).

#### Layers

A stack of 1D Convolutional layers with BatchNorm and ReLU activations.

#### Downsampling

We use a stride of 2 in convolution layers to progressively reduce the temporal resolution while increasing the channel depth.

#### Goal

To learn local temporal patterns (e.g., rising pitch contours, sudden energy bursts) and compress them into a latent space.

### 2. The Pooling Strategy (Temporal Collapse)

To generate a fixed-size embedding from variable-length audio, we use a Hybrid Masked Pooling strategy in the inference phase:

- **Masked Average Pooling:** Calculates the "average vibe" or baseline of the interaction.

- **Masked Max Pooling:** Captures the most salient "peak events" (e.g., highest energy point, most extreme pitch shift).

#### Result

The final embedding is the concatenation of these two vectors, providing a holistic view of both the general tone and specific intensity markers.

### 3. The Decoder (Reconstruction)

#### Layers

Transposed Convolutions (Deconv) that mirror the encoder.

#### Goal

To reconstruct the original frame-by-frame OpenSMILE features from the compressed latent representation.

#### Loss Function

Masked MSE (Mean Squared Error). We compute loss only on the actual audio data, ignoring the zero-padding introduced during batching.

## 🛠️ Data Processing

### Normalization

Neural networks struggle with raw acoustic features because different descriptors have vastly different scales (e.g., Pitch in Hz vs. Jitter as a ratio).

We apply Z-score normalization (`StandardScaler`) using `normalization.FeatureNormalizer`.

Statistics (Mean/Std) are computed globally across the training set and applied consistently during inference.

### Batching

Since audio files vary in duration, we use a custom `pad_collate` function.

Sequences are padded with zeros to match the longest file in the batch.

A binary mask is generated to ensure the model does not learn to reconstruct padding.