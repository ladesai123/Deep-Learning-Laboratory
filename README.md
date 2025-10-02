# Deep Learning Laboratory

This repository contains implementations of various deep learning experiments, covering fundamental to advanced topics in neural networks, computer vision, and autoencoders.

---

## Table of Contents

1. [Experiment 1: Image Classification using Deep Neural Networks (DNN)](#experiment-1-image-classification-using-deep-neural-networks-dnn)
2. [Experiment 3: Object Detection](#experiment-3-object-detection)
3. [Experiment 6: Convolutional Autoencoder](#experiment-6-convolutional-autoencoder)
4. [Experiment 7: Denoising Autoencoder](#experiment-7-denoising-autoencoder)

---

## Experiment 1: Image Classification using Deep Neural Networks (DNN)

**File:** `EXP 1 DL.ipynb`

### Goal
Build a deep neural network (DNN) to classify images into different categories using fully connected layers.

### Procedure

1. **Data Preparation**
   - Load the image dataset from directory structure
   - Normalize pixel values to range [0, 1]
   - Split data into training (80%) and validation (20%) sets
   - Convert images to grayscale for simpler processing

2. **Model Architecture**
   - Flatten input images into 1D vectors
   - Build a multi-layer fully connected neural network
   - Use Dense layers with ReLU activation for hidden layers
   - Apply Dropout for regularization to prevent overfitting
   - Use softmax activation in the output layer for multi-class classification

3. **Training**
   - Compile model with Adam optimizer and categorical crossentropy loss
   - Implement early stopping to prevent overfitting
   - Train for up to 30 epochs with batch size of 32
   - Monitor training and validation accuracy/loss

4. **Evaluation**
   - Generate predictions on test set
   - Visualize sample predictions with true vs predicted labels
   - Calculate accuracy metrics

### Key Libraries
- TensorFlow/Keras
- NumPy
- Matplotlib
- scikit-learn

### Key Concepts
- Fully connected neural networks
- Image preprocessing and normalization
- Training/validation split
- Early stopping
- Dropout regularization

---

## Experiment 3: Object Detection

**Files:** `Exp3ObjDetection.ipynb`, `ImageDetectionExp3.ipynb`

### Goal
Detect and classify objects within images by identifying bounding boxes and object categories.

### Procedure

1. **Dataset Preparation**
   - Load images and corresponding XML annotation files
   - Parse XML files to extract bounding box coordinates and class labels
   - Resize images to a standard size (e.g., 120x120 pixels)
   - Normalize pixel values to [0, 1] range
   - Encode class labels using LabelEncoder

2. **Data Splitting**
   - Split dataset into training (80%) and testing (20%) sets
   - Separate labels and bounding box coordinates for multi-output model

3. **Model Architecture**
   - Build a Convolutional Neural Network (CNN) with:
     - Multiple Conv2D layers for feature extraction
     - MaxPooling2D layers for spatial dimension reduction
     - Flatten layer to convert 2D features to 1D
   - Create two output branches:
     - Classification head: Dense layer with softmax for class prediction
     - Regression head: Dense layer for bounding box coordinates

4. **GPU Configuration**
   - Configure TensorFlow to use specific GPU (cuda:3)
   - Enable GPU memory growth to prevent memory issues

5. **Training**
   - Compile model with separate losses for classification and regression
   - Train the multi-output model
   - Monitor both classification accuracy and bounding box prediction error

6. **Evaluation**
   - Test model on unseen images
   - Visualize predictions with bounding boxes drawn on images
   - Calculate accuracy for classification and IoU for bounding boxes

### Key Libraries
- TensorFlow/Keras
- OpenCV (cv2)
- XML ElementTree
- NumPy
- scikit-learn

### Key Concepts
- Convolutional Neural Networks (CNNs)
- Object detection vs image classification
- Bounding box regression
- Multi-output models
- XML annotation parsing
- GPU configuration and memory management

---

## Experiment 6: Convolutional Autoencoder

**File:** `Exp6ConvAutoEncoder.ipynb`

### Goal
Build a convolutional autoencoder to learn compressed representations of images and reconstruct them, demonstrating unsupervised feature learning.

### Procedure

1. **Data Loading and Preprocessing**
   - Load MNIST dataset (handwritten digits 0-9)
   - Normalize pixel values to [0, 1] range
   - Reshape images to include channel dimension: (28, 28, 1)
   - Use both training (60,000 images) and test (10,000 images) sets

2. **GPU Configuration**
   - Set CUDA_VISIBLE_DEVICES to use specific GPU (cuda:3)
   - Verify GPU availability and configuration
   - Enable proper GPU memory allocation

3. **Encoder Architecture**
   - Input: 28x28x1 grayscale images
   - Conv2D layers with increasing filters (32, 64) for feature extraction
   - MaxPooling2D layers for spatial dimension reduction
   - Progressively reduce spatial dimensions while increasing feature depth
   - Bottleneck: Compressed representation of the input

4. **Decoder Architecture**
   - Conv2DTranspose (upsampling) layers to reverse the encoding process
   - Gradually increase spatial dimensions back to original size
   - Decrease number of filters progressively
   - Final layer: Conv2D with sigmoid activation to output reconstructed image

5. **Training**
   - Compile model with Adam optimizer and binary crossentropy loss
   - Train for 50 epochs with batch size of 256
   - Use validation set to monitor performance
   - Model learns to minimize reconstruction error

6. **Visualization**
   - Display original input images
   - Display reconstructed images from the autoencoder
   - Compare original vs reconstructed to evaluate quality
   - Visualize learned features in the latent space

### Key Libraries
- TensorFlow/Keras
- NumPy
- Matplotlib

### Key Concepts
- Convolutional autoencoders
- Encoder-decoder architecture
- Unsupervised learning
- Feature extraction and reconstruction
- Latent space representation
- Conv2DTranspose (deconvolution/upsampling)
- Image compression and decompression

---

## Experiment 7: Denoising Autoencoder

**File:** `Exp7DenoisingAutoencoder.py`

### Goal
Train a denoising autoencoder that learns to remove noise from corrupted images, reconstructing clean versions from noisy inputs. This demonstrates the autoencoder's ability to learn robust features and perform noise reduction.

### Procedure

1. **Data Loading and Preprocessing**
   - Load MNIST dataset containing 70,000 handwritten digits
     - Training set: 60,000 images
     - Test set: 10,000 images
   - Normalize pixel values from [0, 255] to [0, 1] for training stability
   - Reshape images to (samples, 28, 28, 1) format for Keras compatibility

2. **Creating Noisy Dataset**
   - Add Gaussian noise to clean images to simulate real-world corruption
   - Noise factor: 0.5 (50% intensity)
   - Use `np.random.normal(loc=0.0, scale=1.0)` to generate noise
   - Clip noisy images to valid [0, 1] range to maintain proper pixel values
   - Create noisy versions of both training and test sets

3. **Building the Autoencoder Architecture**
   
   **Input Layer:**
   - Accept 28x28 grayscale images
   
   **Encoder (Compression):**
   - Flatten: Convert 28x28 image to 784-element vector
   - Dense(128, relu): First compression layer (784 → 128)
   - Dense(64, relu): Second compression layer (128 → 64)
   - Dense(32, relu): Bottleneck layer (64 → 32) - compressed representation
   
   **Decoder (Reconstruction):**
   - Dense(64, relu): First expansion layer (32 → 64)
   - Dense(128, relu): Second expansion layer (64 → 128)
   - Dense(784, sigmoid): Final layer (128 → 784) - reconstructed pixels
   - Reshape: Convert 784-element vector back to 28x28 image
   
   **Output:**
   - Sigmoid activation ensures output values are in valid [0, 1] range

4. **Model Compilation**
   - Optimizer: Adam (adaptive learning rate)
   - Loss function: Binary crossentropy (works well for normalized pixel values)
   - The model learns to map: noisy_image → clean_image

5. **Training Process**
   - **Input:** Noisy corrupted images
   - **Target:** Original clean images
   - **Epochs:** 50 (complete passes through the dataset)
   - **Batch size:** 256 images processed simultaneously
   - **Shuffle:** True (randomize training order each epoch)
   - **Validation:** Monitor performance on noisy test set
   - The model learns to identify and remove noise patterns

6. **Denoising and Visualization**
   - Pass noisy test images through the trained autoencoder
   - Generate denoised reconstructions
   - Display comparison grid (10 samples):
     - Row 1: Original clean images
     - Row 2: Corrupted noisy images  
     - Row 3: Denoised reconstructed images
   - Evaluate visual quality of noise removal

### Key Libraries
- **TensorFlow/Keras:** Neural network framework
- **NumPy:** Numerical operations and array handling
- **Matplotlib:** Visualization of results

### Key Concepts
- **Denoising autoencoders:** Neural networks that learn to remove noise
- **Encoder-decoder architecture:** Compression followed by reconstruction
- **Bottleneck layer:** Forced compact representation learns important features
- **Gaussian noise:** Random noise following normal distribution
- **Unsupervised learning:** Learning features without explicit labels
- **Image restoration:** Recovering clean signals from corrupted inputs
- **Sigmoid activation:** Ensures valid pixel value range [0, 1]
- **Binary crossentropy:** Loss function for pixel-wise reconstruction

### Differences from Experiment 6 (Convolutional Autoencoder)
- **Exp 6** uses convolutional layers (spatial feature learning)
- **Exp 7** uses fully connected layers (simpler architecture)
- **Exp 6** focuses on representation learning and compression
- **Exp 7** specifically targets noise removal and image restoration
- Both use encoder-decoder structure but with different layer types

### Expected Results
- The autoencoder should successfully remove most Gaussian noise
- Reconstructed images should closely resemble original clean images
- Some fine details may be lost due to compression through bottleneck
- The model demonstrates robust feature learning despite noisy input

---

## Getting Started

### Prerequisites
```bash
# Install required packages
pip install tensorflow
pip install numpy
pip install matplotlib
pip install opencv-python
pip install scikit-learn
```

### GPU Configuration (Optional but Recommended)
For experiments using GPU (especially Exp 3 and Exp 6):
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Adjust based on your GPU assignment
```

### Running Experiments

**For Jupyter Notebooks (Exp 1, 3, 6):**
1. Open the notebook in Jupyter or JupyterLab
2. Run cells sequentially from top to bottom
3. Ensure GPU configuration is set correctly if using GPU
4. Wait for training to complete
5. View visualizations and results

**For Python Scripts (Exp 7):**
```bash
python Exp7DenoisingAutoencoder.py
```

---

## Repository Structure

```
Deep-Learning-Laboratory/
├── EXP 1 DL.ipynb                    # Experiment 1: DNN Image Classification
├── Exp3ObjDetection.ipynb            # Experiment 3: Object Detection (Version 1)
├── ImageDetectionExp3.ipynb          # Experiment 3: Object Detection (Version 2)
├── Exp6ConvAutoEncoder.ipynb         # Experiment 6: Convolutional Autoencoder
├── Exp7DenoisingAutoencoder.py       # Experiment 7: Denoising Autoencoder
├── Untitled1.ipynb                   # Additional experiments/scratch work
└── README.md                         # This file
```

---

## Notes

- All experiments use TensorFlow/Keras as the primary deep learning framework
- GPU configuration is included in relevant experiments for faster training
- MNIST dataset is used in Experiments 6 and 7 (automatically downloaded by Keras)
- Experiment 1 and 3 require custom datasets to be provided
- Adjust batch sizes and epochs based on your hardware capabilities
- Some experiments may take considerable time to train depending on dataset size

---

## Common Issues and Troubleshooting

### GPU Not Detected
- Verify CUDA and cuDNN installation
- Check GPU assignment with: `nvidia-smi`
- Ensure correct `CUDA_VISIBLE_DEVICES` setting

### Out of Memory Errors
- Reduce batch size
- Use GPU memory growth configuration
- Close other GPU-consuming processes

### Import Errors
- Reinstall required packages
- Verify virtual environment activation
- Check Python version compatibility (Python 3.8+ recommended)

### Dataset Issues
- For MNIST: Will auto-download on first run
- For custom datasets: Ensure proper directory structure and file formats

---

## Future Enhancements

- Add more experiments covering:
  - Recurrent Neural Networks (RNNs)
  - Long Short-Term Memory (LSTM) networks
  - Generative Adversarial Networks (GANs)
  - Transfer Learning with pre-trained models
  - Attention mechanisms and Transformers
- Include pre-trained model weights for faster inference
- Add evaluation metrics and performance benchmarking
- Create Docker container for reproducible environment

---

## License

This repository is intended for educational purposes as part of a Deep Learning Laboratory course.

---

## Contact

For questions or issues, please open an issue in this repository.
