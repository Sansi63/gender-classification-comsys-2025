# Gender Classification - COMSYS Hackathon 2025

## 📌 Overview

This project trains a deep learning model to classify facial images as either **male** or **female**. It is submitted as part of Task A in the COMSYS Hackathon-5.

## 🧠 Model Architecture

- CNN using Keras with Conv2D, MaxPooling, Flatten, and Dense layers.
- Binary classification with sigmoid activation.
- Trained on a dataset structured as:
  
train/

├── male/

└── female/

val/

├── male/

└── female/

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

## 🚀 How to Run

1. **Train the Model**

Run the notebook to train the model. It will automatically save as `gender_model.h5`.

2. **Evaluate the Model**

```bash
python test.py val/
