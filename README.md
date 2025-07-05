# Gender Classification - COMSYS Hackathon 2025 (Task A)

## 📌 Overview

This project trains a deep learning model to classify facial images as either **male** or **female**. It is submitted as part of Task A in the COMSYS Hackathon-5.

## 🧠 Model Architecture

The model is a simple but effective CNN-based binary classifier with the following architecture:

- Input: 224x224 RGB image
- **Conv2D(32 filters, 3x3) + ReLU**
- **MaxPooling2D**
- **Conv2D(64 filters, 3x3) + ReLU**
- **MaxPooling2D**
- **Conv2D(128 filters, 3x3) + ReLU**
- **MaxPooling2D**
- **Flatten**
- **Dense(128, activation='relu')**
- **Dropout(0.5)**
- **Dense(1, activation='sigmoid')**

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

---
  
train/

├── male/

└── female/

val/

├── male/

└── female/

## 📊 Evaluation Metrics
Training:
- **Accuracy: 100%
- **Precision: 100%
- **Recall: 100%
- **F1-Score: 9.80%
Validation:
- **Accuracy: 82.81%
- **Precision: 63.64%
- **Recall: 82.35%
- **F1-Score: 13.59%

## 🚀 How to Run

1. **Train the Model**

Run the notebook to train the model. It will automatically save as `gender_model.h5`.

2. **Evaluate the Model**

```bash 
python test.py val/
```
Face Recognition - COMSYS Hackathon 2025 (Task B)
📌 Overview
This project implements a Siamese neural network for face recognition, matching clear reference images with distorted versions. It is submitted as part of Task B in the COMSYS Hackathon-5, utilizing the DeepFace library for embeddings and a custom Siamese model for similarity learning.

🧠 Model Architecture
- Base Model: Uses VGG-Face from DeepFace to extract 4096-dimensional embeddings from input images.
- Siamese Network:
    1. Two parallel input branches with shared weights.
    2. Embedding layers: Dense(128, activation="relu"), Dense(64, activation="relu").
    3. Output: Euclidean distance between paired embeddings, trained with a contrastive loss function (margin = 1.0).
- Input Shape: (4096,) per embedding.
- Training Data: Structured as train/ and val/ with subdirectories for identities, each containing clear images and a distortion/ subdirectory for distorted images.

📊 Training Results
Training Loss: Decreased from 0.2704 (Epoch 1) to 0.1004 (Epoch 20).
Validation Loss: Decreased from 0.2508 (Epoch 1) to 0.2612 (Epoch 20) over 20 epochs.
Training:
- **Accuracy: 100%
- **Precision: 100%
- **Recall: 100%
- **F1-Score: 63%
Validation:
- **Accuracy: 91%
- **Precision: 88%
- **Recall: 85%
- **F1-Score: 32%
Model Saved: As siamese_face_matching.h5 after training.
🚀 How to Run
1. Install Dependencies
    - Run: pip install deepface tensorflow numpy opencv-python scikit-learn
2. Train the Model
    - Open image-recognition.ipynb.
    -Update TRAIN_DATASET_PATH (e.g., /path/to/train) and VAL_DATASET_PATH (e.g., /path/to/val) to your dataset locations.
    -Run all cells to train the model and save siamese_face_matching.h5.
3. Test the Model
    -Open test-script-img-recog.ipynb.
    -Update VAL_DATASET_PATH (e.g., /path/to/val) and MODEL_PATH (e.g., /path/to/siamese_face_matching.h5) to your validation data and model locations.
    -Run all cells to evaluate the model on the validation set and print accuracy, precision, and recall.
