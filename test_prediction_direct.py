#!/usr/bin/env python
"""Direct test of prediction function without Flask UI"""
import sys
sys.path.insert(0, 'skin_disease_detection')

from app import predict_disease, load_models, CATEGORIES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test 1: Load models
logger.info("="*60)
logger.info("TEST 1: Loading models...")
logger.info("="*60)
if not load_models():
    logger.error("FAILED: Model loading failed")
    sys.exit(1)
logger.info("✓ Models loaded successfully")

# Test 2: Try prediction with a test image
logger.info("\n" + "="*60)
logger.info("TEST 2: Attempting prediction...")
logger.info("="*60)

# Check if we have a sample image to test
import os
test_image = None
for root, dirs, files in os.walk('skin_disease_detection'):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            test_image = os.path.join(root, file)
            break

if test_image:
    logger.info(f"Found test image: {test_image}")
    try:
        disease, confidence = predict_disease(test_image)
        logger.info(f"✓ PREDICTION SUCCESSFUL!")
        logger.info(f"  Disease: {disease}")
        logger.info(f"  Confidence: {confidence}%")
    except Exception as e:
        logger.error(f"✗ PREDICTION FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
else:
    logger.warning("No test image found in project, skipping prediction test")
    logger.info("To test, upload an image via the web interface at http://localhost:5000")

logger.info("\n" + "="*60)
logger.info("TEST COMPLETE")
logger.info("="*60)
