import os
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import logging
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {os.getcwd()}")
print(f"BASE_DIR (app.py location): {BASE_DIR}")

# Define the template directory - THIS IS CRITICAL FOR TEMPLATE NOT FOUND ERROR
# Try multiple possible locations for templates
template_dir = None
possible_template_dirs = [
    # Option 1: templates in ui_components folder at same level as app.py
    os.path.join(BASE_DIR, 'ui_components'),
    # Option 2: templates in templates folder at same level as app.py
    os.path.join(BASE_DIR, 'templates'),
    # Option 3: templates in parent directory's ui_components
    os.path.join(os.path.dirname(BASE_DIR), 'ui_components'),
    # Option 4: templates in parent directory's templates
    os.path.join(os.path.dirname(BASE_DIR), 'templates')
]

# Find the first valid template directory
for possible_dir in possible_template_dirs:
    if os.path.exists(possible_dir):
        template_dir = possible_dir
        break

# If no template directory found, try to create one
if template_dir is None:
    logger.warning("No template directory found. Creating default template directory.")
    template_dir = os.path.join(BASE_DIR, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Check if HTML files exist in current directory and copy them
    for filename in ['index.html', 'result.html', 'report.html', 'error.html']:
        if os.path.exists(filename):
            import shutil
            shutil.copy(filename, os.path.join(template_dir, filename))
            logger.info(f"Copied {filename} to template directory")
    
    # If still no HTML files, create basic ones
    if not any(f.endswith('.html') for f in os.listdir(template_dir)):
        logger.warning("No HTML templates found. Creating minimal templates.")
        
        # Create a minimal index.html
        with open(os.path.join(template_dir, 'index.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Skin Disease Detection</title>
</head>
<body>
    <h1>Upload an image for skin disease detection</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
            ''')
        
        # Create minimal result.html
        with open(os.path.join(template_dir, 'result.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Prediction Result</h1>
    <p>Predicted Disease: {{ prediction }}</p>
    <p>Confidence: {{ confidence }}%</p>
    <a href="/report/{{ disease_id }}">Generate Medical Report</a>
    <a href="/">Upload Another Image</a>
</body>
</html>
            ''')
        
        # Create minimal report.html
        with open(os.path.join(template_dir, 'report.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>{{ disease.name }} Report</title>
</head>
<body>
    <h1>{{ disease.name }} Medical Report</h1>
    <h2>Overview</h2>
    <p>{{ disease.description }}</p>
    <h2>Causes</h2>
    <ul>
    {% for cause in disease.causes %}
        <li>{{ cause }}</li>
    {% endfor %}
    </ul>
    <a href="/">Analyze Another Image</a>
</body>
</html>
            ''')
        
        # Create minimal error.html
        with open(os.path.join(template_dir, 'error.html'), 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
</head>
<body>
    <h1>Error</h1>
    <p>{{ message }}</p>
    <a href="/">Return to Home Page</a>
</body>
</html>
            ''')

# Verify template directory
print(f"Using template directory: {template_dir}")
print("Files in template directory:")
for f in os.listdir(template_dir):
    print(f"  - {f}")

# Initialize Flask app with the correct template folder
app = Flask(__name__,
            template_folder=template_dir,
            static_folder=os.path.join(BASE_DIR, 'static'))

# Configure upload settings
# Use static/uploads so the browser can access images for the result page
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import universal categories and reports
try:
    from skin_disease_detection.categories import CATEGORIES, REPORTS as DISEASE_INFO, FRIENDLY_NAMES
except ImportError:
    from categories import CATEGORIES, REPORTS as DISEASE_INFO, FRIENDLY_NAMES

# Your local model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model_optimized.pkl")
RESNET_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_base_model.h5")

# Global variables for models
svm_model = None
resnet_model = None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image and model.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Unpack if necessary
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # Use a more robust way to index the predicted class
        class_channel = tf.gather(preds, pred_index, axis=1)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # If the conv layer has multiple outputs, ensure we get the tensor
    if isinstance(last_conv_layer_output, (list, tuple)):
        last_conv_layer_output = last_conv_layer_output[0]
        grads = grads[0]

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # and apply ReLU (discard negative activations)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    """
    Overlay the heatmap on the original image and save it.
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    # Modern matplotlib uses colormaps.get_cmap
    try:
        jet = plt.get_cmap("jet")
    except:
        jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    return cam_path

def load_models():
    """Load both SVM and ResNet models with error handling"""
    global svm_model, resnet_model
    try:
        if os.path.exists(SVM_MODEL_PATH):
            with open(SVM_MODEL_PATH, 'rb') as f:
                svm_model = pickle.load(f)
            logger.info("Universal SVM model loaded successfully")
        else:
            logger.error(f"SVM model not found at {SVM_MODEL_PATH}")
            return False

        if os.path.exists(RESNET_MODEL_PATH):
            resnet_model = load_model(RESNET_MODEL_PATH)
            logger.info("ResNet model loaded successfully")
        else:
            logger.error(f"ResNet model not found at {RESNET_MODEL_PATH}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_image(image_path):
    """Preprocess image for ResNet50 (192, 192, with CLAHE)"""
    try:
        image = Image.open(image_path).convert('RGB')
        img = np.array(image)
        img = cv2.resize(img, (192, 192))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_disease(image_path):
    """Predict the disease from an image using the universal model"""
    try:
        processed_img = preprocess_image(image_path)
        features = resnet_model.predict(processed_img, verbose=0)
        features_flat = features.reshape(1, -1).astype(np.float32)
        
        # CalibratedClassifierCV: use predict_proba directly
        probabilities = svm_model.predict_proba(features_flat)[0]
        prediction_idx = int(np.argmax(probabilities))
        confidence = round(float(np.max(probabilities)) * 100, 2)
            
        predicted_label = CATEGORIES[prediction_idx]          # raw folder name
        friendly_label = FRIENDLY_NAMES.get(predicted_label, predicted_label)  # clean display name
        logger.info(f"Predicted: idx={prediction_idx} | folder={predicted_label} | display={friendly_label} | conf={confidence}")
        
        # Generate Grad-CAM heatmap using the ResNet model
        # We target the last convolutional layer
        heatmap = make_gradcam_heatmap(processed_img, resnet_model, "conv5_block3_out")
        
        return friendly_label, confidence, prediction_idx, heatmap
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('error.html', message="No file selected")
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="No file selected")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get prediction and heatmap array
        friendly_label, confidence, disease_idx, heatmap = predict_disease(filepath)
        
        # Save heatmap image
        heatmap_filename = 'heatmap_' + filename
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        save_and_display_gradcam(filepath, heatmap, heatmap_path)
        
        # Generate URLs for the template (using relative paths for static folder)
        original_url = url_for('static', filename='uploads/' + filename)
        heatmap_url = url_for('static', filename='uploads/' + heatmap_filename)
        
        return render_template('result.html',
                               prediction=friendly_label,
                               confidence=confidence,
                               disease_id=disease_idx,
                               original_image_url=original_url,
                               heatmap_image_url=heatmap_url)
    except Exception as e:
        # In case of error, we don't necessarily want to delete if we want to debug, 
        # but for production it's better to cleanup.
        logger.error(f"Error in predict route: {str(e)}")
        return render_template('error.html', message=str(e))

@app.route('/report/<int:disease_id>')
def report(disease_id):
    # disease_id is now a numeric index into CATEGORIES
    if disease_id < 0 or disease_id >= len(CATEGORIES):
        return render_template('error.html', message="Report not found")
    folder_name = CATEGORIES[disease_id]                        # e.g. 'Acne and Rosacea Photos'
    friendly_name = FRIENDLY_NAMES.get(folder_name, folder_name)  # e.g. 'Acne & Rosacea'
    disease = DISEASE_INFO.get(folder_name, {
        "Description": f"Detailed information for {friendly_name}.",
        "Causes": "Various factors including genetics, environment, or infections.",
        "Symptoms": "Localized skin changes, inflammation, or distinctive lesions.",
        "Treatment": "Consult a dermatologist for personalised treatment options."
    })
    
    disease_copy = disease.copy()
    disease_copy['name'] = friendly_name                        # always show clean name
    
    return render_template('report.html', disease=disease_copy, disease_id=disease_id)

@app.route('/error')
def error():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

if __name__ == '__main__':
    # Verify model files exist before trying to load
    if not os.path.exists(SVM_MODEL_PATH):
        logger.error(f"SVM model file not found at {SVM_MODEL_PATH}")
        logger.error("Please check your model path and ensure the file exists")
        logger.error("Current working directory: " + os.getcwd())
        logger.error("BASE_DIR: " + BASE_DIR)
        logger.error("MODEL_DIR: " + MODEL_DIR)
        exit(1)
        
    if not os.path.exists(RESNET_MODEL_PATH):
        logger.error(f"ResNet model file not found at {RESNET_MODEL_PATH}")
        logger.error("Please check your model path and ensure the file exists")
        logger.error("Current working directory: " + os.getcwd())
        logger.error("BASE_DIR: " + BASE_DIR)
        logger.error("MODEL_DIR: " + MODEL_DIR)
        exit(1)
    
    # Load models before starting the server
    if not load_models():
        logger.error("Failed to load models. Application cannot start.")
        logger.error("Possible solutions:")
        logger.error("1. Verify model files exist at the specified paths")
        logger.error("2. Ensure you have enough RAM (at least 4GB free)")
        logger.error("3. Try running with CPU only (already enforced)")
        logger.error("4. Reduce model size or use a smaller model")
        exit(1)
    
    # Run the Flask app
    logger.info("Starting Flask application on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
