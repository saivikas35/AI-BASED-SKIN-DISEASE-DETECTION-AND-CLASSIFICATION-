#!/usr/bin/env python
"""Test prediction API by downloading and uploading a test image"""
import requests
import io
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download a test image (chickenpox from DermNet)
logger.info("="*60)
logger.info("Downloading test image...")
logger.info("="*60)

try:
    # Try to download a skin disease image
    test_urls = [
        "https://www.dermnetnz.org/assets/Uploads/varicella/5/__thumbs/1200x1200__crop/varicella-5.jpg",
        "https://www.dermnetnz.org/assets/Uploads/cellulitis/1/__thumbs/1200x1200__crop/cellulitis-1.jpg",
    ]
    
    img_data = None
    for url in test_urls:
        try:
            logger.info(f"Trying: {url}")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_data = response.content
                logger.info(f"✓ Downloaded successfully ({len(img_data)} bytes)")
                break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {str(e)}")
            continue
    
    if not img_data:
        logger.warning("Could not download test image from web")
        logger.info("Creating a dummy test image locally...")
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        logger.info(f"✓ Created dummy image ({len(img_data)} bytes)")
    
    # Upload to Flask app
    logger.info("\n" + "="*60)
    logger.info("Uploading to Flask app...")
    logger.info("="*60)
    
    files = {'file': ('test.jpg', io.BytesIO(img_data), 'image/jpeg')}
    response = requests.post('http://localhost:5000/predict', files=files, timeout=30)
    
    logger.info(f"Response Status: {response.status_code}")
    
    if response.status_code == 200:
        if 'Error' in response.text or 'error' in response.text.lower():
            logger.error("✗ PREDICTION ERROR (in response HTML)")
            # Extract error message
            if 'Error processing image:' in response.text:
                start = response.text.find('Error processing image:')
                end = response.text.find('</p>', start)
                error_msg = response.text[start:end].replace('Error processing image:', '').strip()
                logger.error(f"  Details: {error_msg}")
        else:
            logger.info("✓ PREDICTION SUCCESSFUL!")
            # Check for disease name in response
            if 'Chickenpox' in response.text or 'Cellulitis' in response.text or 'Ringworm' in response.text:
                logger.info("✓ Disease detected in response")
                # Try to extract confidence
                if 'Confidence:' in response.text or 'confidence' in response.text:
                    logger.info("✓ Confidence score included")
            logger.info(f"Response length: {len(response.text)} bytes")
    else:
        logger.error(f"✗ Server returned status {response.status_code}")
        
except Exception as e:
    logger.error(f"✗ Test failed: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())

logger.info("\n" + "="*60)
logger.info("TEST COMPLETE - Check http://localhost:5000 to upload manually")
logger.info("="*60)
