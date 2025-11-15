# 🌱 Plant Disease Detection System - Complete Technical Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [CNN Architecture - Exact Specifications](#cnn-architecture---exact-specifications)
3. [Training Methodology](#training-methodology)
4. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
5. [Hyperparameters](#hyperparameters)
6. [Testing & Evaluation](#testing--evaluation)
7. [Model Specifications](#model-specifications)
8. [Inference Pipeline](#inference-pipeline)

---

## 📊 Project Overview

**Project Name**: Plant Disease Detection System  
**Framework**: PyTorch 1.8.1  
**Language**: Python 3.9+  
**Architecture**: Custom 4-Layer Convolutional Neural Network (CNN)  
**Task**: Multi-class Plant Disease Classification  
**Dataset**: Plant Village Dataset + New Plant Diseases Dataset (Kaggle)  
**Total Classes**: 39-42+ (dynamic, based on dataset)  

---

## 🏗️ CNN Architecture - Exact Specifications

### Model Architecture Overview
- **Type**: Custom 4-Convolutional Layer CNN
- **Depth**: 4 convolutional blocks + 2 fully connected layers
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Normalization**: Batch Normalization (after each conv layer)
- **Regularization**: Dropout (0.4) in fully connected layers
- **Pooling**: Max Pooling (2x2 kernel, stride=2)

### Detailed Layer-by-Layer Architecture

#### **Input Layer**
- **Input Shape**: `[Batch, 3, 224, 224]` (RGB images)
- **Preprocessing**: Resize(255) → CenterCrop(224) → ToTensor()

---

#### **Convolutional Block 1 (Conv1)**
```
Layer 1: Conv2d
  - Input Channels: 3
  - Output Channels: 32
  - Kernel Size: 3×3
  - Padding: 1
  - Stride: 1 (default)
  - Parameters: (3×3×3 + 1) × 32 = 896 parameters

Layer 2: ReLU Activation

Layer 3: BatchNorm2d
  - Features: 32

Layer 4: Conv2d
  - Input Channels: 32
  - Output Channels: 32
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×32 + 1) × 32 = 9,248 parameters

Layer 5: ReLU Activation

Layer 6: BatchNorm2d
  - Features: 32

Layer 7: MaxPool2d
  - Kernel Size: 2×2
  - Stride: 2
  - Output Shape: [Batch, 32, 112, 112]
```

**Total Parameters in Conv1 Block**: ~10,144 + BatchNorm params

---

#### **Convolutional Block 2 (Conv2)**
```
Layer 1: Conv2d
  - Input Channels: 32
  - Output Channels: 64
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×32 + 1) × 64 = 18,496 parameters

Layer 2: ReLU Activation

Layer 3: BatchNorm2d
  - Features: 64

Layer 4: Conv2d
  - Input Channels: 64
  - Output Channels: 64
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×64 + 1) × 64 = 36,928 parameters

Layer 5: ReLU Activation

Layer 6: BatchNorm2d
  - Features: 64

Layer 7: MaxPool2d
  - Kernel Size: 2×2
  - Stride: 2
  - Output Shape: [Batch, 64, 56, 56]
```

**Total Parameters in Conv2 Block**: ~55,424 + BatchNorm params

---

#### **Convolutional Block 3 (Conv3)**
```
Layer 1: Conv2d
  - Input Channels: 64
  - Output Channels: 128
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×64 + 1) × 128 = 73,856 parameters

Layer 2: ReLU Activation

Layer 3: BatchNorm2d
  - Features: 128

Layer 4: Conv2d
  - Input Channels: 128
  - Output Channels: 128
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×128 + 1) × 128 = 147,584 parameters

Layer 5: ReLU Activation

Layer 6: BatchNorm2d
  - Features: 128

Layer 7: MaxPool2d
  - Kernel Size: 2×2
  - Stride: 2
  - Output Shape: [Batch, 128, 28, 28]
```

**Total Parameters in Conv3 Block**: ~221,440 + BatchNorm params

---

#### **Convolutional Block 4 (Conv4)**
```
Layer 1: Conv2d
  - Input Channels: 128
  - Output Channels: 256
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×128 + 1) × 256 = 295,168 parameters

Layer 2: ReLU Activation

Layer 3: BatchNorm2d
  - Features: 256

Layer 4: Conv2d
  - Input Channels: 256
  - Output Channels: 256
  - Kernel Size: 3×3
  - Padding: 1
  - Parameters: (3×3×256 + 1) × 256 = 590,080 parameters

Layer 5: ReLU Activation

Layer 6: BatchNorm2d
  - Features: 256

Layer 7: MaxPool2d
  - Kernel Size: 2×2
  - Stride: 2
  - Output Shape: [Batch, 256, 14, 14]
```

**Total Parameters in Conv4 Block**: ~885,248 + BatchNorm params

---

#### **Flatten Layer**
```
Input Shape: [Batch, 256, 14, 14]
Output Shape: [Batch, 50176]
Flattened Size: 256 × 14 × 14 = 50,176 features
```

---

#### **Fully Connected Layers (Dense)**

**FC Layer 1:**
```
Layer 1: Dropout
  - Dropout Rate: 0.4
  - Probability of keeping: 0.6

Layer 2: Linear
  - Input Features: 50,176
  - Output Features: 1,024
  - Parameters: (50,176 × 1,024) + 1,024 = 51,382,784 parameters

Layer 3: ReLU Activation

Layer 4: Dropout
  - Dropout Rate: 0.4
```

**FC Layer 2 (Output Layer):**
```
Layer 1: Linear
  - Input Features: 1,024
  - Output Features: K (number of classes, typically 39)
  - Parameters: (1,024 × K) + K = 1,025K parameters
  - For 39 classes: (1,024 × 39) + 39 = 39,975 parameters
```

---

### **Total Model Parameters**

**Convolutional Layers**: ~1,172,256 parameters  
**Fully Connected Layers**: ~51,422,759 parameters (for 39 classes)  
**BatchNorm Parameters**: ~1,024 parameters (4 layers × 256 avg)  
**Total Trainable Parameters**: ~52,595,039 parameters (~52.6M parameters)  

**Model Size**: ~200.66 MB (as saved .pt file)

---

## 🎓 Training Methodology

### Dataset Split
- **Training Set**: 70% of total dataset
- **Validation Set**: 15% of total dataset
- **Test Set**: 15% of total dataset
- **Split Method**: Stratified split (maintains class distribution)
- **Random State**: 42 (for reproducibility)

### Data Augmentation Factor
- **Training Augmentation**: Each image replicated 10 times during training
- **Effective Training Size**: Original training set × 10
- **Purpose**: Increases dataset size for better generalization with limited data

### Training Loop Structure
```
For each epoch (1 to 50):
  1. Training Phase:
     - Set model to training mode
     - Forward pass through network
     - Compute loss (CrossEntropyLoss)
     - Backward pass (gradient computation)
     - Update weights (Adam optimizer)
     - Track training loss and accuracy
  
  2. Validation Phase:
     - Set model to evaluation mode
     - Forward pass (no gradients)
     - Compute validation loss and accuracy
     - Save model if validation accuracy improves
  
  3. Learning Rate Scheduling:
     - Reduce learning rate by factor of 0.1 every 10 epochs
```

---

## 🔧 Data Preprocessing & Augmentation

### **Training Transform Pipeline**
```python
transforms.Compose([
    transforms.Resize(255),              # Resize to 255 pixels (largest side)
    transforms.RandomCrop(224),          # Randomly crop 224×224 patch
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance horizontal flip
    transforms.RandomRotation(degrees=15),   # Rotate ±15 degrees
    transforms.ColorJitter(
        brightness=0.2,                  # ±20% brightness variation
        contrast=0.2,                    # ±20% contrast variation
        saturation=0.2                   # ±20% saturation variation
    ),
    transforms.ToTensor()                # Convert to tensor [0,1] range
])
```

### **Validation/Test Transform Pipeline**
```python
transforms.Compose([
    transforms.Resize(255),              # Resize to 255 pixels
    transforms.CenterCrop(224),          # Center crop 224×224
    transforms.ToTensor()                # Convert to tensor [0,1] range
])
```

**No augmentation for validation/test** - ensures fair evaluation

### **Inference Transform (Production)**
```python
transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
```

---

## ⚙️ Hyperparameters

### **Training Hyperparameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 32 | Images per batch |
| **Learning Rate** | 0.001 | Initial learning rate |
| **Epochs** | 50 | Maximum training epochs |
| **Optimizer** | Adam | Adaptive moment estimation optimizer |
| **Loss Function** | CrossEntropyLoss | Multi-class classification loss |
| **Learning Rate Scheduler** | StepLR | Step size: 10, Gamma: 0.1 |
| **Weight Initialization** | PyTorch Default | Kaiming/He initialization for conv layers |

### **Learning Rate Schedule**
- **Initial LR**: 0.001
- **After Epoch 10**: 0.0001 (LR × 0.1)
- **After Epoch 20**: 0.00001 (LR × 0.01)
- **After Epoch 30**: 0.000001 (LR × 0.001)
- **After Epoch 40**: 0.0000001 (LR × 0.0001)

### **Regularization Techniques**

1. **Batch Normalization**
   - Applied after each convolutional layer
   - Normalizes activations to reduce internal covariate shift
   - Accelerates training and improves stability

2. **Dropout**
   - Rate: 0.4 (40% neurons randomly deactivated)
   - Applied in fully connected layers only
   - Prevents overfitting by reducing co-adaptation

3. **Data Augmentation**
   - Random cropping, flipping, rotation, color jittering
   - Increases effective dataset size
   - Improves generalization to unseen data

---

## 📈 Testing & Evaluation

### **Test Set Evaluation**

After training completes, the model is evaluated on the held-out test set:

```python
Test Metrics:
  - Test Accuracy: Calculated as (correct_predictions / total_predictions) × 100
  - Top-1 Accuracy: Single highest probability prediction
  - Confusion Matrix: Not explicitly computed (can be added)
```

### **Validation Metrics (During Training)**

For each epoch:
- **Training Loss**: Average CrossEntropyLoss on training set
- **Training Accuracy**: Percentage of correct predictions on training set
- **Validation Loss**: Average CrossEntropyLoss on validation set
- **Validation Accuracy**: Percentage of correct predictions on validation set
- **Best Model**: Saved when validation accuracy improves

### **Model Selection Strategy**

- **Criterion**: Best validation accuracy
- **Checkpoint Saving**: Automatically saves model when validation accuracy improves
- **Final Model**: Model with highest validation accuracy across all epochs

### **Expected Performance**

Based on typical CNN performance on Plant Village dataset:
- **Training Accuracy**: 90-98%
- **Validation Accuracy**: 85-95%
- **Test Accuracy**: 85-95%

*Note: Actual performance depends on dataset size and quality*

---

## 🔍 Model Specifications

### **Input Specifications**
- **Image Format**: RGB (3 channels)
- **Image Size**: 224×224 pixels
- **Input Range**: [0, 1] (normalized)
- **Batch Dimension**: Variable (typically 1 for inference)

### **Output Specifications**
- **Output Type**: Logits (raw scores before softmax)
- **Output Shape**: [Batch, K] where K = number of classes
- **Softmax Applied**: During inference (in `app.py`)
- **Prediction**: Argmax of softmax output (highest probability class)
- **Confidence Score**: Maximum softmax probability

### **Feature Map Dimensions Through Network**

```
Input:        [B, 3, 224, 224]
After Conv1:  [B, 32, 112, 112]   (MaxPool reduces by 2×)
After Conv2:  [B, 64, 56, 56]     (MaxPool reduces by 2×)
After Conv3:  [B, 128, 28, 28]    (MaxPool reduces by 2×)
After Conv4:  [B, 256, 14, 14]    (MaxPool reduces by 2×)
After Flatten: [B, 50176]
After FC1:    [B, 1024]
After FC2:    [B, K]              (K = number of classes)
```

---

## 🚀 Inference Pipeline

### **Prediction Process**

1. **Image Upload**: User uploads image via Flask web interface
2. **Preprocessing**:
   ```python
   - Open image with PIL/Pillow
   - Convert to RGB if needed
   - Apply transform: Resize(255) → CenterCrop(224) → ToTensor()
   - Add batch dimension: unsqueeze(0)
   ```
3. **Model Inference**:
   ```python
   - Set model to eval() mode
   - Forward pass (no gradients)
   - Get raw logits output
   - Apply softmax to get probabilities
   ```
4. **Prediction Extraction**:
   ```python
   - Get class index: argmax(probabilities)
   - Get confidence: max(probabilities)
   - Get top-3 predictions (optional)
   ```
5. **Result Display**:
   - Disease name from class mapping
   - Description from disease_info.csv
   - Prevention steps from disease_info.csv
   - Recommended supplement from supplement_info.csv

### **Inference Code Flow**

```python
# From app.py - prediction() function
def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    input_data = transform(image)           # [3, 224, 224]
    input_data = input_data.unsqueeze(0)    # [1, 3, 224, 224]
    
    with torch.no_grad():
        output = model(input_data)          # [1, K] logits
        output = F.softmax(output, dim=1)   # [1, K] probabilities
        output = output.detach().numpy()
    
    index = np.argmax(output)               # Predicted class index
    confidence = float(np.max(output))      # Confidence score
    
    return index
```

---

## 📊 Class Information

### **Current Classes (39 Original)**

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Background_without_leaves
6. Blueberry___healthy
7. Cherry___Powdery_mildew
8. Cherry___healthy
9. Corn___Cercospora_leaf_spot Gray_leaf_spot
10. Corn___Common_rust
11. Corn___Northern_Leaf_Blight
12. Corn___healthy
13. Grape___Black_rot
14. Grape___Esca_(Black_Measles)
15. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
16. Grape___healthy
17. Orange___Haunglongbing_(Citrus_greening)
18. Peach___Bacterial_spot
19. Peach___healthy
20. Pepper,_bell___Bacterial_spot
21. Pepper,_bell___healthy
22. Potato___Early_blight
23. Potato___Late_blight
24. Potato___healthy
25. Raspberry___healthy
26. Soybean___healthy
27. Squash___Powdery_mildew
28. Strawberry___Leaf_scorch
29. Strawberry___healthy
30. Tomato___Bacterial_spot
31. Tomato___Early_blight
32. Tomato___Late_blight
33. Tomato___Leaf_Mold
34. Tomato___Septoria_leaf_spot
35. Tomato___Spider_mites Two-spotted_spider_mite
36. Tomato___Target_Spot
37. Tomato___Tomato_Yellow_Leaf_Curl_Virus
38. Tomato___Tomato_mosaic_virus
39. Tomato___healthy

*Note: Dataset currently has 42 classes (3 new classes added)*

---

## 🛠️ Training Script Details

### **File**: `Model/train_model.py`

**Key Functions:**
1. `load_dataset()`: Loads images from directory structure
2. `augment_dataset()`: Replicates images for augmentation (factor=10)
3. `PlantDiseaseDataset`: Custom PyTorch Dataset class
4. `train_model()`: Main training function

**Training Flow:**
```
1. Load dataset → Get image paths and labels
2. Split dataset → Train/Val/Test (70/15/15)
3. Augment training data → 10× replication
4. Create DataLoaders → Batch size 32
5. Initialize model → CNN(K classes)
6. Define optimizer → Adam, LR=0.001
7. Define scheduler → StepLR, step=10, gamma=0.1
8. Training loop (50 epochs):
   - Train on training set
   - Validate on validation set
   - Save best model (based on val accuracy)
9. Evaluate on test set
10. Save class mappings to JSON
```

---

## 📝 Model File Information

### **Saved Model Format**
- **Format**: PyTorch state_dict (.pt file)
- **Size**: ~200.66 MB
- **Location**: `Model/plant_disease_model_1_latest.pt`
- **Contents**: Model weights only (not full model architecture)
- **Loading**: `torch.load(model_path, map_location='cpu')`

### **Class Mappings**
- **Format**: JSON file
- **Location**: `Model/class_mappings.json`
- **Contents**:
  ```json
  {
    "class_to_idx": {"Apple___Apple_scab": 0, ...},
    "idx_to_class": {"0": "Apple___Apple_scab", ...},
    "num_classes": 39
  }
  ```

---

## 🔬 Performance Characteristics

### **Computational Requirements**

**Training:**
- **Device**: CPU or CUDA-enabled GPU
- **Memory**: ~2-4 GB RAM (CPU), ~4-8 GB VRAM (GPU)
- **Time**: 
  - CPU: ~30-60 minutes for 50 epochs (depends on dataset size)
  - GPU: ~5-15 minutes for 50 epochs

**Inference:**
- **Device**: CPU (optimized for CPU deployment)
- **Memory**: ~500 MB RAM
- **Time**: ~0.1-0.5 seconds per image (CPU)

### **Model Efficiency**
- **Parameters**: ~52.6M (medium-sized CNN)
- **FLOPs**: ~1.2 GFLOPs per inference
- **Model Size**: 200.66 MB (compressed weights)

---

## 📚 Additional Information

### **Dataset Sources**
1. **Primary**: Plant Village Dataset (original)
2. **Extended**: New Plant Diseases Dataset from Kaggle
   - Dataset ID: `vipoooool/new-plant-diseases-dataset`
   - Augmented dataset with multiple images per class

### **Supported Image Formats**
- JPG/JPEG
- PNG
- Case-insensitive (.jpg, .JPG, .jpeg, .JPEG, .png, .PNG)

### **Framework Versions**
- PyTorch: 1.8.1+cpu
- Torchvision: 0.9.1+cpu
- NumPy: 1.20.2
- Pillow: 8.2.0
- Pandas: 1.2.4
- Scikit-learn: (for train_test_split)

---

## 🎯 Summary

This project implements a **custom 4-layer CNN** for plant disease classification with:
- **52.6M parameters** across convolutional and fully connected layers
- **Data augmentation** for improved generalization
- **Batch normalization** and **dropout** for regularization
- **Adam optimizer** with **learning rate scheduling**
- **70/15/15 train/val/test split** for robust evaluation
- **Expected accuracy**: 85-95% on test set

The model is designed for **real-time inference** on CPU, making it suitable for deployment in resource-constrained environments like mobile devices or edge computing systems.

---

**Document Version**: 1.0  
**Last Updated**: Based on current codebase analysis  
**Author**: Technical Documentation Generated from Code Analysis

