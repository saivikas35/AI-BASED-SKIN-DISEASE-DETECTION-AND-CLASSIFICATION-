import numpy as np
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = r"d:\Dup\Advance-Skin-disease-diagnosis-using-Image-processing"
X_PATH = os.path.join(BASE_DIR, "all_features_X.npy")
Y_PATH = os.path.join(BASE_DIR, "all_features_Y.npy")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "svm_model_optimized.pkl")

def train_universal():
    logger.info("--- Starting Universal SVC Training ---")
    
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        logger.error("Features not found! Ensure extraction is complete.")
        return

    # Load features
    logger.info("Loading features...")
    X = np.load(X_PATH).astype(np.float32)
    Y = np.load(Y_PATH)
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features each.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Train Linear SVC with Calibration for probabilities
    logger.info("Training Linear SVC (this may take a few minutes)...")
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    
    base_clf = LinearSVC(C=1.0, dual=False, max_iter=2000, random_state=42)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Universal Model Accuracy: {acc*100:.2f}%")
    
    # Classification Report
    try:
        import sys
        sys.path.append(os.path.join(BASE_DIR, 'skin_disease_detection'))
        from categories import CATEGORIES
    except ImportError:
        CATEGORIES = [f"Class_{i}" for i in range(23)]
        
    logger.info("\n" + classification_report(y_test, y_pred, target_names=CATEGORIES))

    # Save model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    logger.info(f"Universal model saved to {MODEL_SAVE_PATH}")
    logger.info("Training complete!")

if __name__ == "__main__":
    train_universal()
