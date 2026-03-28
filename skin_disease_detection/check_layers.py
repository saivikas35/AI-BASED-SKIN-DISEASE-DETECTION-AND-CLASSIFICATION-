import tensorflow as tf
import os

BASE_DIR = r'd:\Dup\Advance-Skin-disease-diagnosis-using-Image-processing\skin_disease_detection'
RESNET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'resnet50_base_model.h5')

if os.path.exists(RESNET_MODEL_PATH):
    model = tf.keras.models.load_model(RESNET_MODEL_PATH)
    print("Model layers:")
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name or 'out' in layer.name:
            print(f"{i}: {layer.name}")
else:
    print(f"Model not found at {RESNET_MODEL_PATH}")
