"""
Plant Disease Detection API
Flask backend with PyTorch ResNet34 model
For BTech Final Year Project
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'model', 'plantDisease-resnet34.pth')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')
DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.json')

# Load class indices
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices_dict = json.load(f)
    DISEASE_CLASSES = [class_indices_dict[str(i)] for i in range(len(class_indices_dict))]
    NUM_CLASSES = len(DISEASE_CLASSES)
    logger.info(f"Loaded {NUM_CLASSES} disease classes")
except Exception as e:
    logger.error(f"Failed to load class indices: {e}")
    DISEASE_CLASSES = []
    NUM_CLASSES = 38

# Load disease information
try:
    with open(DISEASE_INFO_PATH, 'r') as f:
        DISEASE_INFO = json.load(f)
    logger.info(f"Loaded disease information for {len(DISEASE_INFO)} conditions")
except Exception as e:
    logger.error(f"Failed to load disease info: {e}")
    DISEASE_INFO = {}


def load_model(model_path, num_classes):
    """Load PyTorch ResNet34 model from .pth file"""
    try:
        # Create a ResNet34 model
        model = models.resnet34(weights=None)
        
        # Modify the final layer to match number of disease classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'network.' prefix if present (model was saved with wrapper)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('network.'):
                new_key = key.replace('network.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")


# Load model
try:
    model = load_model(MODEL_PATH, NUM_CLASSES)
    model_loaded = True
except Exception as e:
    logger.warning(f"Model loading failed: {e}")
    model = None
    model_loaded = False


# Image preprocessing transforms
def get_transforms():
    """Get image preprocessing transforms matching training"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

preprocess = get_transforms()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = preprocess(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def format_disease_name(disease_class):
    """Format disease class name for display"""
    # Replace underscores and clean up
    formatted = disease_class.replace('___', ' - ').replace('_', ' ')
    # Handle special cases
    formatted = formatted.replace('(', '(').replace(')', ')')
    return formatted


def get_disease_info(disease_class):
    """Get detailed disease information from the database"""
    if disease_class in DISEASE_INFO:
        return DISEASE_INFO[disease_class]
    
    # Return default info if not found
    return {
        "disease_name": format_disease_name(disease_class),
        "plant": disease_class.split('___')[0].replace('_', ' ') if '___' in disease_class else "Unknown",
        "scientific_name": None,
        "description": "Detailed information not available for this condition.",
        "symptoms": [],
        "causes": [],
        "treatment": ["Consult with a local agricultural expert for proper diagnosis and treatment."],
        "prevention": ["Maintain good plant hygiene", "Monitor plants regularly"],
        "severity": "unknown",
        "is_contagious": False
    }


def get_severity_color(severity):
    """Get color code for severity level"""
    colors = {
        "none": "#22c55e",      # Green
        "low": "#84cc16",       # Lime
        "moderate": "#eab308",  # Yellow
        "high": "#f97316",      # Orange
        "critical": "#ef4444",  # Red
        "unknown": "#6b7280"    # Gray
    }
    return colors.get(severity, colors["unknown"])


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "🌿 Plant Disease Detection API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model_loaded,
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "endpoints": {
            "predict": "POST /predict - Upload image for disease prediction",
            "health": "GET /health - Check API health status",
            "diseases": "GET /diseases - Get list of detectable diseases",
            "disease_info": "GET /disease/<name> - Get info about specific disease"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "classes_available": NUM_CLASSES
    })


@app.route('/diseases')
def get_diseases():
    """Get list of all detectable diseases"""
    diseases = []
    for disease_class in DISEASE_CLASSES:
        info = get_disease_info(disease_class)
        diseases.append({
            "id": disease_class,
            "name": info["disease_name"],
            "plant": info["plant"],
            "severity": info["severity"],
            "is_healthy": "healthy" in disease_class.lower()
        })
    return jsonify({
        "total": len(diseases),
        "diseases": diseases
    })


@app.route('/disease/<disease_id>')
def get_disease_details(disease_id):
    """Get detailed information about a specific disease"""
    if disease_id not in DISEASE_INFO:
        return jsonify({"error": "Disease not found"}), 404
    
    info = DISEASE_INFO[disease_id]
    return jsonify({
        "id": disease_id,
        **info,
        "severity_color": get_severity_color(info.get("severity", "unknown"))
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if not model_loaded:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please check server logs."
            }), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Please upload an image."
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No image file selected."
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Read and preprocess the image
        try:
            image = Image.open(io.BytesIO(file.read()))
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Could not read image file. Please upload a valid image."
            }), 400
        
        # Preprocess
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predictions
        probabilities = probabilities.cpu().numpy()[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_index])
        
        # Get predicted disease class
        if predicted_class_index < len(DISEASE_CLASSES):
            predicted_class = DISEASE_CLASSES[predicted_class_index]
        else:
            predicted_class = "Unknown"
        
        # Get disease information
        disease_info = get_disease_info(predicted_class)
        
        # Determine if healthy
        is_healthy = "healthy" in predicted_class.lower()
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            if idx < len(DISEASE_CLASSES):
                disease_class = DISEASE_CLASSES[idx]
                info = get_disease_info(disease_class)
                top_predictions.append({
                    "disease_id": disease_class,
                    "disease_name": info["disease_name"],
                    "plant": info["plant"],
                    "confidence": float(probabilities[idx]),
                    "confidence_percentage": round(float(probabilities[idx]) * 100, 2),
                    "is_healthy": "healthy" in disease_class.lower()
                })
        
        # Build response
        response = {
            "success": True,
            "prediction": {
                "disease_id": predicted_class,
                "disease_name": disease_info["disease_name"],
                "plant": disease_info["plant"],
                "scientific_name": disease_info.get("scientific_name"),
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2),
                "is_healthy": is_healthy,
                "severity": disease_info["severity"],
                "severity_color": get_severity_color(disease_info["severity"])
            },
            "disease_info": {
                "description": disease_info["description"],
                "symptoms": disease_info["symptoms"],
                "causes": disease_info["causes"],
                "treatment": disease_info["treatment"],
                "prevention": disease_info["prevention"],
                "is_contagious": disease_info["is_contagious"]
            },
            "top_predictions": top_predictions,
            "metadata": {
                "model_version": "ResNet34-v1",
                "timestamp": datetime.now().isoformat(),
                "device": str(device)
            }
        }
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.route('/stats')
def get_stats():
    """Get API statistics"""
    # Count healthy vs diseased
    healthy_count = sum(1 for d in DISEASE_CLASSES if 'healthy' in d.lower())
    diseased_count = len(DISEASE_CLASSES) - healthy_count
    
    # Count by plant
    plants = {}
    for disease in DISEASE_CLASSES:
        plant = disease.split('___')[0].replace('_', ' ') if '___' in disease else "Unknown"
        if plant not in plants:
            plants[plant] = {"total": 0, "healthy": 0, "diseases": 0}
        plants[plant]["total"] += 1
        if "healthy" in disease.lower():
            plants[plant]["healthy"] += 1
        else:
            plants[plant]["diseases"] += 1
    
    return jsonify({
        "total_classes": len(DISEASE_CLASSES),
        "healthy_classes": healthy_count,
        "disease_classes": diseased_count,
        "plants": plants,
        "model_info": {
            "architecture": "ResNet34",
            "input_size": "224x224",
            "device": str(device)
        }
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 16MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         🌿 Plant Disease Detection API v2.0 🌿           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Endpoints:                                              ║
    ║  • GET  /          - API info                            ║
    ║  • GET  /health    - Health check                        ║
    ║  • GET  /diseases  - List all diseases                   ║
    ║  • GET  /disease/<id> - Disease details                  ║
    ║  • POST /predict   - Upload image for prediction         ║
    ║  • GET  /stats     - API statistics                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
