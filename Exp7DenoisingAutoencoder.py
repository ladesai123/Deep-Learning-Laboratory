"""
Experiment 7: Denoising Autoencoder
====================================
This experiment demonstrates a denoising autoencoder that learns to reconstruct
clean images from noisy versions using the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Step 1: Load and Preprocess the MNIST Dataset
# MNIST contains 70,000 handwritten digits (0-9) - 60k training, 10k test
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values from [0,255] to [0,1] for better training stability
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape data to include channel dimension: (samples, 28, 28, 1)
# This format is required by Keras for grayscale images
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Step 2: Create Noisy Versions of the Images (Simulating Real-World Corruption)
noise_factor = 0.5  # Controls how much noise to add (50% intensity)

# Add Gaussian noise (random noise following normal distribution)
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to ensure they remain in valid [0,1] range for images
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Step 3: Build the Denoising Autoencoder Model Architecture
# Define input layer accepting 28x28 pixel grayscale images
input_img = Input(shape=(28, 28, 1))

# ENCODER: Compresses the noisy image into a smaller representation
# Flatten 28x28 image into 784-element vector
x = Flatten()(input_img)

# Encoding layers (progressively smaller, capturing important features)
encoded = Dense(128, activation='relu')(x)  # First compression: 784 -> 128
encoded = Dense(64, activation='relu')(encoded)   # Second compression: 128 -> 64
encoded = Dense(32, activation='relu')(encoded)   # Final encoding: 64 -> 32 (bottleneck)

# DECODER: Reconstructs clean image from compressed representation
decoded = Dense(64, activation='relu')(encoded)   # First expansion: 32 -> 64
decoded = Dense(128, activation='relu')(decoded)  # Second expansion: 64 -> 128
decoded = Dense(28 * 28, activation='sigmoid')(decoded)  # Final layer: 128 -> 784
# Sigmoid activation ensures output values between 0-1 (valid pixel values)

# Reshape 784-element vector back to 28x28 image
decoded = Reshape((28, 28, 1))(decoded)

# Create the autoencoder model that maps noisy input to clean output
autoencoder = Model(input_img, decoded)

# Compile the model with Adam optimizer and binary crossentropy loss
# Binary crossentropy works well for pixel values between 0-1
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Step 4: Train the Model
# The model learns to reconstruct clean images from noisy versions
# Input: noisy images, Target: original clean images
autoencoder.fit(x_train_noisy, x_train,  # Learn: noisy → clean mapping
                epochs=50,               # Train for 50 complete passes through dataset
                batch_size=256,          # Process 256 images at once
                shuffle=True,            # Shuffle training data each epoch
                validation_data=(x_test_noisy, x_test))  # Evaluate on test data

# Step 5: Generate Denoised Images using Trained Model
# Feed noisy test images through autoencoder to get cleaned versions
denoised_images = autoencoder.predict(x_test_noisy)

# Step 6: Visualize the Results - Compare Original vs Noisy vs Denoised
n = 10  # Number of comparison images to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Display original clean images (row 1)
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Display noisy input images (row 2)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Display denoised/reconstructed images (row 3)
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.show()
