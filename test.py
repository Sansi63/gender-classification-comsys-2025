# test.py

import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def evaluate_model(test_path):
    model = load_model("gender_model.h5")
    
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    y_true = test_gen.classes
    y_pred = model.predict(test_gen)
    y_pred = (y_pred > 0.5).astype(int).reshape(-1)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_test_data>")
        sys.exit(1)

    evaluate_model(sys.argv[1])