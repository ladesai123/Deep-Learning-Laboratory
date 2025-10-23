# Updated: Ex No 11 - GAN (simplified)
# - Reduced default epochs for demo
# - Fixed label shapes and noise sampling
# - Added graceful stop, model saving, and loss saving

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)

# Load MNIST and preprocess to [-1, 1]
mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = (x_train / 255.0) * 2.0 - 1.0
x_test  = (x_test  / 255.0) * 2.0 - 1.0

# Flatten images
N, H, W = x_train.shape
D = H * W
x_train = x_train.reshape(-1, D)
x_test  = x_test.reshape(-1, D)

latent_dim = 100

def build_generator(latent_dim, img_dim=D):
    i = Input(shape=(latent_dim,))
    x = Dense(256)(i)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(img_dim, activation='tanh')(x)
    return Model(i, x, name='generator')

def build_discriminator(img_dim=D):
    i = Input(shape=(img_dim,))
    x = Dense(512)(i)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(i, x, name='discriminator')

# Build models
generator = build_generator(latent_dim)
discriminator = build_discriminator(D)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Combined model (for generator training)
z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
fake_pred = discriminator(img)
combined = Model(z, fake_pred)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training params
batch_size = 32
epochs = 5000  # reduced for demo; increase if desired
sample_period = 500
ones = np.ones((batch_size, 1))
zeros = np.zeros((batch_size, 1))

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

def sample_images(step):
    rows, cols = 5, 5
    noise = tf.random.normal((rows * cols, latent_dim))
    imgs = generator.predict(noise, verbose=0)
    imgs = 0.5 * imgs + 0.5
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx].reshape(H, W), cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    fig.savefig(f"gan_images/{step}.png")
    plt.close(fig)

d_losses = []
g_losses = []

try:
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        noise = tf.random.normal((batch_size, latent_dim))
        fake_imgs = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        # Train generator (via combined)
        noise = tf.random.normal((batch_size, latent_dim))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, ones)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.4f}, d_acc: {d_acc:.4f}, g_loss: {g_loss:.4f}")

        if (epoch + 1) % sample_period == 0:
            sample_images(epoch + 1)

except KeyboardInterrupt:
    print("Training interrupted by user. Saving progress...")

# Save models and losses
generator.save_weights('saved_models/generator_weights.h5')
discriminator.save_weights('saved_models/discriminator_weights.h5')
np.save('saved_models/d_losses.npy', np.array(d_losses))
np.save('saved_models/g_losses.npy', np.array(g_losses))

print("Finished. Models and losses saved to saved_models/")
