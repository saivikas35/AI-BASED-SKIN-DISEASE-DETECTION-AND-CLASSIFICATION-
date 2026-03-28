import tensorflow as tf
import os

MODEL_PATH = r'd:\Dup\Advance-Skin-disease-diagnosis-using-Image-processing\skin_disease_detection\models\resnet50_base_model.h5'
if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
print("\nLast few layers:")
for layer in model.layers[-10:]:
    print(f"{layer.name}: {layer.output_shape}")
