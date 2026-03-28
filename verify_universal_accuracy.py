import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = r"d:\Dup\Advance-Skin-disease-diagnosis-using-Image-processing"
X_PATH = os.path.join(BASE_DIR, "all_features_X.npy")
Y_PATH = os.path.join(BASE_DIR, "all_features_Y.npy")
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model_optimized.pkl")

def verify():
    print("Loading features...")
    X = np.load(X_PATH).astype(np.float32)
    y = np.load(Y_PATH)
    
    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
        
    print("Evaluating...")
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFINAL UNIVERSAL 23-CLASS ACCURACY: {acc*100:.2f}%")
    
    try:
        import sys
        sys.path.append(os.path.join(BASE_DIR, 'skin_disease_detection'))
        from categories import CATEGORIES
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    except:
        print("\nClassification Report (Generic Labels):")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    verify()
