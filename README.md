# Digit-Recognizer
# Multi-Digit Handwritten Number Recognition with CRNN

This repository contains a deep learning implementation of **multi-digit handwritten number recognition** using a **Convolutional Recurrent Neural Network (CRNN)**. The model is built using **TensorFlow** and **Keras**, leveraging a **CNN (Convolutional Neural Network)** as an encoder and **Bidirectional LSTM (Long Short-Term Memory)** as a decoder, trained on the **MNIST dataset**. 

# CRNN-based OCR for Multi-Digit Recognition using CTC Loss

This repository implements an Optical Character Recognition (OCR) model using a Convolutional Recurrent Neural Network (CRNN) architecture with Connectionist Temporal Classification (CTC) loss. The model is trained to recognize sequences of three consecutive digits, built on the MNIST dataset.

---

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation and Prediction](#evaluation-and-prediction)
- [Concepts Used](#concepts-used)
- [Why the Specific Techniques Were Used](#why-the-specific-techniques-were-used)
- [Examples](#examples)

---

## Overview

The code trains a CRNN model to recognize sequences of three consecutive digits (e.g., `123`) using the MNIST dataset. By stacking images of digits horizontally, the model learns to recognize digit sequences in images of shape `28x84` (three MNIST digits side-by-side). This approach is designed for OCR tasks where sequence prediction is required. The CTC loss function allows the model to output variable-length predictions without the need for fixed alignment between input and output.

### Key Features

- **CRNN Architecture**: A Convolutional Neural Network (CNN) extracts spatial features from the input images, and Bidirectional LSTMs (Bi-LSTMs) are used to model the sequence information, allowing the network to predict multiple digits at once.
  
- **CTC Loss**: The model is trained using **CTC loss**, a popular loss function for sequence-to-sequence tasks where the alignment between input and output is unknown. This makes it ideal for handwriting recognition where the sequence of digits must be predicted directly from the image.

- **Multi-Digit Recognition**: The model can predict sequences of digits (such as "123") from a single image, making it suitable for tasks like automatic number plate recognition (ANPR) or reading multi-digit handwritten numbers.

- **Data Augmentation**: Real-time augmentation techniques such as rotation, zoom, and shifts help increase the robustness of the model by training on more diverse variations of the data, improving its generalization on unseen test data.

---

## Setup

### Requirements

- Python 3.7+
- TensorFlow (2.x)
- NumPy
- Matplotlib
  
## Dataset Preparation
The code loads the MNIST dataset and then preprocesses it to create images of three-digit sequences. Each sequence is created by stacking three consecutive MNIST digit images horizontally.
```python
def load_multi_digit_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalization and stacking steps
    return x_train_multi, y_train_multi, x_test_multi, y_test_multi
```

The function `load_multi_digit_mnist` returns processed `x_train`, `y_train`, `x_test`, and `y_test` datasets, where each image is a concatenation of three digits.

## Model Architecture
The CRNN model architecture consists of:

- **Convolutional Neural Network (CNN)** layers for feature extraction.
- **Bidirectional LSTM** layers for sequence modeling.
- **CTC Loss layer** for handling alignment between input and output sequences.

```python
def build_crnn_model(input_shape=(28, 28*3, 1), num_classes=10):
    # Define CNN layers
    # Define Bi-directional LSTM layers
    # Define Dense layer and CTC Loss layer
    return model, prediction_model
```

- **CNN Layers**: Extract spatial features from images.
- **Bi-directional LSTM Layers**: Learn dependencies in the sequence.
- **Dense Layer**: Outputs probabilities for each digit class.
- **CTC Loss Layer**: Allows sequence alignment between input images and predicted digit sequences.

The code defines both a model for training with CTC loss and a `prediction_model` for inference.

## Training the Model
The model is trained using a data generator (`CTCAugmentedDataGenerator`) that performs data augmentation and provides additional inputs required for CTC loss:

```python
train_gen = CTCAugmentedDataGenerator(x_train, y_train, batch_size=32)
epochs = 8

model.fit(
    train_gen,
    epochs=epochs,
    validation_data=(
        [x_test_tensor, y_test_tensor, input_length_test_tensor, label_length_test_tensor],
        tf.convert_to_tensor(np.zeros(len(x_test)), dtype=tf.float32),
    ),
)
```

- **Batch Size**: Set to 32.
- **Epochs**: 8 (can be adjusted for accuracy vs. training time).
- **Validation Data**: Preprocessed test data with CTC-specific inputs.

## Evaluation and Prediction
The model is evaluated on test data, and predictions are made using the `prediction_model`. CTC decoding is applied to interpret the predicted sequences of digits.

```python
def decode_predictions(preds):
    # Decode CTC predictions
    return tf.keras.backend.get_value(decoded)

test_preds = prediction_model.predict(x_test)
decoded_preds = decode_predictions(test_preds)
```

The predictions are displayed alongside the test images for visual validation.

## Concepts Used
1. **Convolutional Neural Networks (CNNs)**: Specialized for processing spatial data. CNN layers extract spatial features (edges, shapes) from each image.

2. **Recurrent Neural Networks (RNNs) and LSTMs**: Used for sequence data, with Bi-directional LSTM layers capturing forward and backward dependencies in digit sequences.

3. **Connectionist Temporal Classification (CTC)**: Allows sequence prediction where the length of the input does not match the output, ideal for OCR tasks.

4. **Data Augmentation**: Improves generalization by applying rotations, shifts, and zoom transformations, creating more varied training data.

## Why the Specific Techniques Were Used

### 1. **Convolutional Neural Networks (CNNs)**
   - CNNs are effective for image processing tasks and spatial feature extraction, crucial for recognizing digit shapes in images.

### 2. **Bidirectional LSTM (Bi-LSTM)**
   - LSTM layers capture temporal dependencies, and Bi-LSTM layers provide both past and future context, improving multi-digit sequence recognition.

### 3. **Connectionist Temporal Classification (CTC) Loss**
   - CTC loss allows sequence prediction without precise alignment, ideal for multi-digit OCR tasks where segmentation is challenging.

### 4. **Data Augmentation**
   - Augmenting data simulates real-world variations, improving model robustness.

### 5. **Multi-Digit Sequence Generation**
   - Stacking MNIST images horizontally enables recognition of multi-digit numbers, useful for applications like license plate or postal code recognition.

## Examples
Below are examples of predicted sequences alongside input images. Each image shows a sequence of three digits, and the predicted output is displayed above.

```python
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28 * 3), cmap="gray")
    plt.title(f"Predicted: {''.join(map(str, decoded_preds[i]))}")
    plt.show()
```

## Summary
This project demonstrates a CRNN-based OCR model for multi-digit recognition using CTC loss. The model learns to recognize digit sequences in images and can generalize to varying sequence lengths, thanks to CTC. It is well-suited for OCR tasks where sequence prediction is required, such as reading license plates or text in natural images.

