# COMP6252 Coursework 1 - Music Genre Classification

## 📋 Project Overview

This project implements six different neural network architectures to classify music genres using the GTZAN dataset. The goal is to compare the performance of various deep learning approaches on audio classification tasks.

**Module:** COMP6252  
**Due Date:** Tuesday 28th April, 16:00  
**Weight:** 20% of overall module mark  
**Submission:** https://handin.ecs.soton.ac.uk/handin/2526/COMP6252/1

---

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
cd COMP6252-CW1

# Install required packages
pip install torch torchvision torchaudio
pip install librosa numpy matplotlib pandas jupyter

# Download GTZAN dataset from Kaggle
# Place in ./data/gtzan/ directory
```

---

## 📁 Project Structure

```
COMP6252-CW1/
│
├── README.md                   # This file
├── notebook.ipynb              # Main Jupyter notebook with all implementations
├── report.pdf                  # Final 4-page report (3 pages + 1 page reflection)
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
└── code.zip                    # Zipped code for submission
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

### Step 4: Generate Report
- Document implementation details
- Present performance metrics
- Discuss and compare results
- Write reflection on deep learning topic

---

## 📊 Expected Deliverables

### 1. Code (code.zip)
- Single Jupyter notebook containing:
  - Data loading and preprocessing
  - All 6 neural network architectures (Net1-Net6)
  - Training loops for each architecture
  - Evaluation and comparison code
  - Results visualization

### 2. Report (report.pdf)
**Part 1 (3 pages):**
- Name and ECS user ID
- Implementation description
  - Network architectures
  - Training procedures
  - Hyperparameters used
- Performance results
  - Accuracy, loss curves
  - Confusion matrices
  - Comparative metrics
- Discussion and analysis
  - Which architectures performed best/worst?
  - Why did certain approaches work better?
  - Insights and observations

**Part 2 (1 page):**
- Selected deep learning topic from labs
- Reflection covering:
  - Importance of the topic
  - Current technologies
  - Personal implementation capabilities
  - Positive/negative impacts and future vision

**Format:** CVPR LaTeX template, maximum 4 A4 pages, no appendix

---

## 🎯 Marking Criteria

**Total: 20 marks (16 + 4)**

- **Task Completion:** Successful implementation of all 6 architectures
- **Understanding:** Clear evidence of comprehension
- **Code Quality:** Well-structured, commented, professional
- **Report Quality:** Clear presentation, insightful analysis
- **Professionalism:** Overall quality of implementation and reporting

---

## 📝 Implementation Tips

### For Architectures 1-4 (Image-based)
1. Use `torchvision.transforms.Resize(180)` for image preprocessing
2. Use `torch.utils.data.random_split()` for dataset splitting
3. Monitor both training and validation loss/accuracy
4. Save models with best validation performance

### For Architectures 5-6 (Audio-based)
1. Use librosa for audio feature extraction
2. Consider using MFCC features or raw waveforms
3. Implement early stopping based on validation loss
4. For Architecture 6, train GAN separately first, then use for augmentation

### General Best Practices
- Use GPU if available (`model.to(device)`)
- Implement learning rate scheduling
- Save checkpoints regularly
- Visualize training progress
- Document all hyperparameters

---

## 🔗 Useful Resources

- **Module Website:** https://secure.ecs.soton.ac.uk/module/2526/COMP6252/43095/
- **Jupyter Notebook Docs:** https://docs.jupyter.org/en/latest/
- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html
- **Librosa Documentation:** https://librosa.org/doc/latest/index.html

---

## 📧 Contact

For questions or issues:
- Use Q&A channel on Teams
- Email: Hikmat, Zhiwu, or Xiaohao

---

## ⚠️ Important Notes

- **Late Submission:** Standard ECS penalties apply
- **Academic Integrity:** This is individual work
- **Submission Format:** 
  - `report.pdf` (CVPR format, max 4 pages)
  - `code.zip` (all code files)
- **Deadline:** Tuesday 28th April, 16:00

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
- [ ] Write report Part 1 (3 pages)
- [ ] Write report Part 2 (1 page reflection)
- [ ] Create code.zip
- [ ] Submit to ECS Handin before deadline

---

**Good luck with your coursework! 🎓**
