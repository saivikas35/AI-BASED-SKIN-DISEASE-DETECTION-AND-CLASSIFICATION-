#!/usr/bin/env python3
"""
Debug API prediction with proper file upload
"""
import requests
import logging
from PIL import Image
import numpy as np
import os
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Create a dummy test image
test_image_path = 'test_dummy.jpg'
dummy_array = np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8)
img = Image.fromarray(dummy_array)
img.save(test_image_path)
logger.info(f"✓ Created test image: {test_image_path} ({os.path.getsize(test_image_path)} bytes)")

# Upload to Flask API
url = 'http://127.0.0.1:5000/predict'
try:
    with open(test_image_path, 'rb') as f:
        # Use the correct parameter name ('file' not 'image')
        files = {'file': ('test_dummy.jpg', f, 'image/jpeg')}
        response = requests.post(url, files=files, timeout=30)
    
    logger.info(f"Response Status: {response.status_code}")
    
    # Extract error from HTML if present
    html_content = response.text
    
    # Write full response to file for inspection
    with open('test_response.html', 'w') as f:
        f.write(html_content)
    logger.info(f"✓ Response written to test_response.html ({len(html_content)} bytes)")
    
    logger.info("\n" + "="*60)
    logger.info("RESPONSE (checking for errors):")
    logger.info("="*60)
    
    if 'error' in html_content.lower() or 'traceback' in html_content.lower():
        # Extract just the error message portion
        if '<div class="error-message">' in html_content:
            error_start = html_content.find('<div class="error-message">') + len('<div class="error-message">')
            error_end = html_content.find('</div>', error_start)
            error_msg = html_content[error_start:error_end].strip()
            logger.error(f"ERROR: {error_msg}")
        else:
            logger.info(html_content[:1500])
    else:
        logger.info("✓ SUCCESS - Prediction result page received!")
        # Try to extract the disease name and confidence
        if 'class="prediction"' in html_content:
            import re
            pred_match = re.search(r'class="prediction">(.*?)</div>', html_content, re.DOTALL)
            if pred_match:
                disease = pred_match.group(1).strip()
                logger.info(f"✓ Disease predicted: {disease}")
        if 'Confidence' in html_content:
            conf_match = re.search(r'Confidence:.*?<strong>(.*?)</strong>', html_content, re.DOTALL)
            if conf_match:
                confidence = conf_match.group(1).strip()
                logger.info(f"✓ Confidence: {confidence}")
        logger.info("\n✓✓✓ FULL PREDICTION PIPELINE WORKING! ✓✓✓")
    
except Exception as e:
    logger.error(f"Request failed: {e}")
    import traceback
    traceback.print_exc()
