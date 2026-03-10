# COMP6252 Coursework 1 - Music Genre Classification
## 📋 Project Overview

This project implements six different neural network architectures to classify music genres using the GTZAN dataset. The goal is to compare the performance of various deep learning approaches on audio classification tasks.

## 🎵 Dataset

**GTZAN Music Genre Dataset**
- **Source:** [Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Genres:** 10 different music genres (jazz, classical, rock, blues, country, disco, hiphop, metal, pop, reggae)
- **Format:** Audio samples with corresponding MEL spectrogram visualizations
- **Total Samples:** ~1000 audio files (100 per genre)

### Data Split
- **Training:** 70%
- **Validation:** 20%
- **Test:** 10%

---

## 🏗️ Neural Network Architectures

### Architecture 1: Fully Connected Network
- **Type:** Basic Neural Network
- **Structure:** Input → Hidden Layer 1 → Hidden Layer 2 → Output
- **Input:** MEL spectrogram images (180×180)
- **Epochs:** 50 and 100

### Architecture 2: Convolutional Neural Network (CNN)
- **Type:** CNN (based on Figure 1 in coursework brief)
- **Layers:** 
  - Conv1 → Conv2 → MaxPool1 → Conv3 → Conv4 → MaxPool2 → Fully Connected → Output
- **Input:** MEL spectrogram images (180×180)
- **Activation:** ReLU
- **Epochs:** 50 and 100

### Architecture 3: CNN with Batch Normalization
- **Type:** Enhanced CNN
- **Modification:** Adds Batch Normalization layer to Architecture 2
- **Benefit:** Improved training stability and convergence
- **Input:** MEL spectrogram images (180×180)
- **Epochs:** 50 and 100

### Architecture 4: CNN with RMSProp Optimizer
- **Type:** CNN with Batch Normalization
- **Modification:** Uses RMSProp optimizer instead of default (e.g., Adam/SGD)
- **Benefit:** Alternative optimization strategy
- **Input:** MEL spectrogram images (180×180)
- **Epochs:** 50 and 100

### Architecture 5: RNN with LSTMs
- **Type:** Recurrent Neural Network
- **Structure:** LSTM layers for sequence processing
- **Input:** Audio samples (raw audio or audio features)
- **Benefit:** Captures temporal dependencies in music
- **Epochs:** Train until convergence (monitor validation loss)

### Architecture 6: RNN with GANs for Data Augmentation
- **Type:** RNN with LSTM + Generative Adversarial Network
- **Modification:** GANs generate synthetic audio samples to augment training data
- **Augmentation:** Generate equal number of synthetic samples as original training data
- **Input:** Audio samples
- **Epochs:** Train until convergence

---

## 🛠️ Setup and Installation

### Prerequisites
```bash
# Python 3.8+
# PyTorch
# torchvision
# librosa (for audio processing)
# numpy
# matplotlib
# pandas
# jupyter
```

### Installation
```bash
# Clone or download the project
cd Music-Genre-Classification-ML-models-CNN-and-RNN-with-LSTMs

# Install required packages
pip install torch torchvision torchaudio
pip install librosa numpy matplotlib pandas jupyter

# Download GTZAN dataset from Kaggle
# Place in ./data/gtzan/ directory
```

---

## 📁 Project Structure

```
Music-Genre-Classification-ML-models-CNN-and-RNN-with-LSTMs/
│
├── README.md                   # This file
├── music-genre-classification.ipynb              # Main Jupyter notebook with all implementations
│
├── data/
│   └── gtzan/                  # GTZAN dataset
│       ├── images_original/    # MEL spectrogram images
│       └── genres_original/    # Audio files
│
├── models/                     # Saved model checkpoints
│   ├── net1_model.pth
│   ├── net2_model.pth
│   ├── net3_model.pth
│   ├── net4_model.pth
│   ├── net5_model.pth
│   └── net6_model.pth
│
├── results/                    # Training results and plots
│   ├── training_curves/
│   ├── confusion_matrices/
│   └── performance_metrics.csv
│

```

---

## 🚀 How to Run

### Step 1: Prepare the Data
```python
# In the Jupyter notebook
# Load and preprocess GTZAN dataset
# Resize images to 180×180 for architectures 1-4
# Split into train/validation/test sets
```

### Step 2: Train Models
```python
# Define all 6 architectures (Net1, Net2, Net3, Net4, Net5, Net6)

# For Net1-Net4:
# - Train for 50 epochs
# - Train for 100 epochs
# - Record performance metrics

# For Net5-Net6:
# - Train until convergence
# - Use early stopping based on validation loss
```

### Step 3: Evaluate and Compare
```python
# Test all models on the test set
# Generate confusion matrices
# Calculate accuracy, precision, recall, F1-score
# Compare performance across architectures
```



---
###  Architectures 1-4 (Image-based)
1. Used `torchvision.transforms.Resize(180)` for image preprocessing
2. Used `torch.utils.data.random_split()` for dataset splitting
3. Monitored both training and validation loss/accuracy
4. Saved models with best validation performance

###  Architectures 5-6 (Audio-based)
1. Used librosa for audio feature extraction
2. Considered using MFCC features or raw waveforms
3. Implemented early stopping based on validation loss
4. For Architecture 6, trained GAN separately first, then used for augmentation



---

## 🔗 Useful Resources

- **Module Website:** https://secure.ecs.soton.ac.uk/module/2526/COMP6252/43095/
- **Jupyter Notebook Docs:** https://docs.jupyter.org/en/latest/
- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html
- **Librosa Documentation:** https://librosa.org/doc/latest/index.html

---

## 📈 Performance Tracking Template

| Architecture | 50 Epochs Accuracy | 100 Epochs Accuracy | Test Accuracy | Notes |
|--------------|-------------------|---------------------|---------------|-------|
| Net1 (FC)    |                   |                     |               |       |
| Net2 (CNN)   |                   |                     |               |       |
| Net3 (BN)    |                   |                     |               |       |
| Net4 (RMSProp)|                  |                     |               |       |
| Net5 (RNN)   | N/A               | N/A                 |               |       |
| Net6 (GAN)   | N/A               | N/A                 |               |       |

---

## 🏆 Success Checklist

- [ ] Download GTZAN dataset
- [ ] Set up Python environment with all dependencies
- [ ] Implement Net1 (Fully Connected)
- [ ] Implement Net2 (CNN)
- [ ] Implement Net3 (CNN + Batch Norm)
- [ ] Implement Net4 (CNN + RMSProp)
- [ ] Implement Net5 (RNN + LSTM)
- [ ] Implement Net6 (RNN + GAN)
- [ ] Train all models and record results
- [ ] Generate performance visualizations


