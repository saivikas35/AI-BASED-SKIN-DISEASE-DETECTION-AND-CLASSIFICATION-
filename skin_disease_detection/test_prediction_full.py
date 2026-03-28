import os
import numpy as np
import tensorflow as tf
from app import load_models, predict_disease, save_and_display_gradcam, resnet_model, svm_model

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(BASE_DIR, 'test_dummy.jpg')

# Create a dummy image if it doesn't exist
if not os.path.exists(TEST_IMAGE):
    from PIL import Image
    img = Image.new('RGB', (192, 192), color='red')
    img.save(TEST_IMAGE)

def test_full_flow():
    try:
        print("Loading models...")
        if not load_models():
            print("Failed to load models")
            return
        
        print(f"Testing prediction for {TEST_IMAGE}...")
        friendly_label, confidence, disease_idx, heatmap = predict_disease(TEST_IMAGE)
        print(f"Result: {friendly_label} ({confidence}%)")
        
        heatmap_path = os.path.join(BASE_DIR, 'static', 'uploads', 'test_heatmap.jpg')
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        
        print("Generating Grad-CAM heatmap...")
        save_and_display_gradcam(TEST_IMAGE, heatmap, heatmap_path)
        print(f"Heatmap saved to {heatmap_path}")
        
    except Exception as e:
        print(f"Caught error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_full_flow()
