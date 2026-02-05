"""
Plant Disease Detection API
Flask backend with PyTorch Multi-Model Support
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
from typing import Dict, Any, Optional

# Load .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed - using system environment variables only")

# MongoDB support (optional)
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ pymongo not installed - using hardcoded model registry only")

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
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'model')
# NEW (Correct)
# MODEL_DIR = os.path.join(BASE_DIR, 'model')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')
DISEASE_INFO_PATH = os.path.join(BASE_DIR, 'disease_info.json')

# ============================================
# MONGODB CONFIGURATION
# ============================================
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.environ.get('MONGO_DB', 'plantguard')
MONGO_COLLECTION = os.environ.get('MONGO_COLLECTION', 'models')

# MongoDB connection (lazy initialization)
mongo_client = None
mongo_db = None
mongo_collection = None

def get_mongo_collection():
    """Get MongoDB collection with lazy initialization"""
    global mongo_client, mongo_db, mongo_collection
    
    if not MONGODB_AVAILABLE:
        return None
    
    if mongo_collection is not None:
        return mongo_collection
    
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Test connection
        mongo_client.server_info()
        mongo_db = mongo_client[MONGO_DB]
        mongo_collection = mongo_db[MONGO_COLLECTION]
        logger.info(f"✓ Connected to MongoDB: {MONGO_DB}.{MONGO_COLLECTION}")
        return mongo_collection
    except Exception as e:
        logger.warning(f"⚠️ MongoDB connection failed: {e}")
        logger.warning("⚠️ Using hardcoded MODEL_REGISTRY as fallback")
        return None


# ============================================
# HARDCODED MODEL REGISTRY (FALLBACK)
# ============================================
# This is used if MongoDB is unavailable
# Keep this updated as a backup!

HARDCODED_MODEL_REGISTRY = {
    "Resnet34":{
        "file": "PlantCareModelV2.0.pth",
        "architecture": "resnet34",
        "name": "ResNet-34 V1.0",
        "description": "Custom trained model by Archit using ResNet-34 architecture",
        "accuracy": "98.420%",
        "input_size": 128,
        "is_default": True
    },
    "Resnet34_Adv":{
        "file": "plantDisease-resnet34adv.pth",
        "architecture": "resnet34",
        "name": "ResNet-34 V2.0",
        "description": "Custom trained model by Archit using ResNet-34 architecture",
        "accuracy": "98.420%",
        "input_size": 128,
        "is_default": False
    },
}


def get_model_registry() -> Dict[str, Any]:
    """
    Get model registry from MongoDB, fallback to hardcoded if unavailable.
    Returns dict of model_id -> model_config
    """
    collection = get_mongo_collection()
    
    if collection is None:
        logger.info("Using hardcoded MODEL_REGISTRY")
        return HARDCODED_MODEL_REGISTRY.copy()
    
    try:
        # Fetch all models from MongoDB
        models = {}
        for doc in collection.find({"enabled": {"$ne": False}}):
            model_id = doc.get("model_id") or doc.get("_id")
            if isinstance(model_id, str):
                models[model_id] = {
                    "file": doc.get("file"),
                    "architecture": doc.get("architecture", "resnet34"),
                    "name": doc.get("name", model_id),
                    "description": doc.get("description", ""),
                    "accuracy": doc.get("accuracy", "N/A"),
                    "input_size": doc.get("input_size", 128),
                    "is_default": doc.get("is_default", False)
                }
        
        if models:
            logger.info(f"Loaded {len(models)} models from MongoDB")
            return models
        else:
            logger.warning("MongoDB returned no models, using hardcoded fallback")
            return HARDCODED_MODEL_REGISTRY.copy()
            
    except Exception as e:
        logger.error(f"Error fetching from MongoDB: {e}")
        return HARDCODED_MODEL_REGISTRY.copy()


# Initialize MODEL_REGISTRY from MongoDB or fallback
MODEL_REGISTRY = get_model_registry()

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


# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

def create_model_architecture(architecture: str, num_classes: int) -> nn.Module:
    """Create model architecture based on type"""
    
    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "resnet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "resnet101":
        model = models.resnet101(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "vgg16":
        model = models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "vgg19":
        model = models.vgg19(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model


def load_model_weights(model: nn.Module, model_path: str) -> nn.Module:
    """Load weights into model, handling different checkpoint formats"""
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove common prefixes if present
    new_state_dict = {}
    prefixes_to_remove = ['network.', 'model.', 'module.']
    
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                break
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    return model


def load_single_model(model_id: str, model_config: Dict[str, Any]) -> Optional[nn.Module]:
    """Load a single model from the registry"""
    try:
        model_path = os.path.join(MODEL_DIR, model_config["file"])
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        # Create architecture
        model = create_model_architecture(model_config["architecture"], NUM_CLASSES)
        
        # Load weights
        model = load_model_weights(model, model_path)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"✓ Loaded model: {model_config['name']} ({model_id})")
        return model
        
    except Exception as e:
        logger.error(f"✗ Failed to load model {model_id}: {e}")
        return None


# ============================================
# LOADED MODELS STORAGE
# ============================================

loaded_models: Dict[str, nn.Module] = {}
default_model_id: Optional[str] = None

def initialize_models():
    """Initialize all models from the registry"""
    global loaded_models, default_model_id
    
    logger.info("=" * 50)
    logger.info("Loading models...")
    logger.info("=" * 50)
    
    for model_id, config in MODEL_REGISTRY.items():
        model = load_single_model(model_id, config)
        if model is not None:
            loaded_models[model_id] = model
            if config.get("is_default", False):
                default_model_id = model_id
    
    # If no default set, use first available model
    if default_model_id is None and loaded_models:
        default_model_id = list(loaded_models.keys())[0]
    
    logger.info("=" * 50)
    logger.info(f"Loaded {len(loaded_models)}/{len(MODEL_REGISTRY)} models")
    if default_model_id:
        logger.info(f"Default model: {default_model_id}")
    logger.info("=" * 50)

# Initialize models on startup
initialize_models()

# For backward compatibility
model_loaded = len(loaded_models) > 0


def is_model_loaded():
    """Check if any models are loaded - dynamic check"""
    return len(loaded_models) > 0


# Image preprocessing transforms - MUST MATCH TRAINING!
# Training used: transforms.Compose([transforms.Resize(size=128), transforms.ToTensor()])
def get_transforms():
    """Get image preprocessing transforms matching training"""
    return transforms.Compose([
        transforms.Resize(128),  # Training used 128x128, NOT 224!
        transforms.ToTensor()    # Training had NO normalization!
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
        "models_loaded": len(loaded_models),
        "available_models": list(loaded_models.keys()),
        "default_model": default_model_id,
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "endpoints": {
            "predict": "POST /predict - Upload image for disease prediction",
            "models": "GET /models - List all available models",
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
        "model_loaded": is_model_loaded(),
        "models_available": len(loaded_models),
        "default_model": default_model_id,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "classes_available": NUM_CLASSES
    })


@app.route('/models')
def get_models():
    """Get list of all available models"""
    models_list = []
    
    for model_id, config in MODEL_REGISTRY.items():
        is_loaded = model_id in loaded_models
        models_list.append({
            "id": model_id,
            "name": config["name"],
            "architecture": config["architecture"],
            "description": config["description"],
            "accuracy": config.get("accuracy", "N/A"),
            "input_size": config.get("input_size", 224),
            "is_default": config.get("is_default", False),
            "is_loaded": is_loaded,
            "status": "ready" if is_loaded else "not_found"
        })
    
    return jsonify({
        "success": True,
        "total_registered": len(MODEL_REGISTRY),
        "total_loaded": len(loaded_models),
        "default_model": default_model_id,
        "models": models_list
    })


@app.route('/models/<model_id>')
def get_model_info(model_id):
    """Get detailed information about a specific model"""
    if model_id not in MODEL_REGISTRY:
        return jsonify({"error": f"Model '{model_id}' not found in registry"}), 404
    
    config = MODEL_REGISTRY[model_id]
    is_loaded = model_id in loaded_models
    
    return jsonify({
        "id": model_id,
        "name": config["name"],
        "architecture": config["architecture"],
        "description": config["description"],
        "accuracy": config.get("accuracy", "N/A"),
        "input_size": config.get("input_size", 224),
        "is_default": config.get("is_default", False),
        "is_loaded": is_loaded,
        "status": "ready" if is_loaded else "not_found",
        "file": config["file"]
    })


# ============================================
# MODEL MANAGEMENT API (MongoDB)
# ============================================

@app.route('/admin/models', methods=['POST'])
def add_model():
    """Add a new model to MongoDB registry"""
    collection = get_mongo_collection()
    if collection is None:
        return jsonify({"success": False, "error": "MongoDB not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON data provided"}), 400
    
    required_fields = ["model_id", "file", "architecture", "name"]
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
    
    model_doc = {
        "model_id": data["model_id"],
        "file": data["file"],
        "architecture": data.get("architecture", "resnet34"),
        "name": data["name"],
        "description": data.get("description", ""),
        "accuracy": data.get("accuracy", "N/A"),
        "input_size": data.get("input_size", 128),
        "is_default": data.get("is_default", False),
        "enabled": True,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Upsert (update if exists, insert if not)
        collection.update_one(
            {"model_id": data["model_id"]},
            {"$set": model_doc},
            upsert=True
        )
        return jsonify({"success": True, "message": f"Model '{data['model_id']}' added/updated", "model": model_doc})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/admin/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete (disable) a model from MongoDB registry"""
    collection = get_mongo_collection()
    if collection is None:
        return jsonify({"success": False, "error": "MongoDB not available"}), 503
    
    try:
        # Soft delete - just disable it
        result = collection.update_one(
            {"model_id": model_id},
            {"$set": {"enabled": False, "deleted_at": datetime.now().isoformat()}}
        )
        
        if result.modified_count > 0:
            return jsonify({"success": True, "message": f"Model '{model_id}' disabled"})
        else:
            return jsonify({"success": False, "error": f"Model '{model_id}' not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/admin/models/<model_id>', methods=['PUT'])
def update_model(model_id):
    """Update a model in MongoDB registry"""
    collection = get_mongo_collection()
    if collection is None:
        return jsonify({"success": False, "error": "MongoDB not available"}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON data provided"}), 400
    
    # Only allow updating certain fields
    allowed_fields = ["file", "architecture", "name", "description", "accuracy", "input_size", "is_default", "enabled"]
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    update_data["updated_at"] = datetime.now().isoformat()
    
    try:
        result = collection.update_one(
            {"model_id": model_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return jsonify({"success": True, "message": f"Model '{model_id}' updated"})
        else:
            return jsonify({"success": False, "error": f"Model '{model_id}' not found or no changes"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/admin/reload', methods=['POST'])
def reload_models():
    """Reload all models from MongoDB (hot reload without restart)"""
    global MODEL_REGISTRY, loaded_models, default_model_id
    
    try:
        # Refresh registry from MongoDB
        MODEL_REGISTRY = get_model_registry()
        
        # Unload all current models
        loaded_models.clear()
        default_model_id = None
        
        # Reload models
        initialize_models()
        
        return jsonify({
            "success": True,
            "message": "Models reloaded successfully",
            "models_loaded": len(loaded_models),
            "registry_count": len(MODEL_REGISTRY),
            "default_model": default_model_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/admin/status')
def admin_status():
    """Get admin status including MongoDB connection"""
    collection = get_mongo_collection()
    
    return jsonify({
        "mongodb_available": MONGODB_AVAILABLE,
        "mongodb_connected": collection is not None,
        "mongo_uri": MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI,  # Hide credentials
        "mongo_db": MONGO_DB,
        "mongo_collection": MONGO_COLLECTION,
        "using_fallback": collection is None,
        "registry_source": "mongodb" if collection else "hardcoded",
        "models_in_registry": len(MODEL_REGISTRY),
        "models_loaded": len(loaded_models),
        "default_model": default_model_id
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
    """Main prediction endpoint - supports model selection"""
    try:
        if not is_model_loaded():
            return jsonify({
                "success": False,
                "error": "No models loaded. Please check server logs."
            }), 500
        
        # Get selected model (from form data or query param)
        selected_model_id = request.form.get('model') or request.args.get('model') or default_model_id
        
        # Validate model selection
        if selected_model_id not in loaded_models:
            available = list(loaded_models.keys())
            return jsonify({
                "success": False,
                "error": f"Model '{selected_model_id}' not available. Available models: {available}"
            }), 400
        
        # Get the selected model
        model = loaded_models[selected_model_id]
        model_config = MODEL_REGISTRY[selected_model_id]
        
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
        
        # Check uncertainty - if confidence below threshold, mark as uncertain
        UNCERTAINTY_THRESHOLD = 0.85  # 85%
        is_uncertain = confidence < UNCERTAINTY_THRESHOLD
        
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
            "is_uncertain": is_uncertain,
            "uncertainty_message": "Low confidence - this plant may not be in our training data. Results may be inaccurate." if is_uncertain else None,
            "prediction": {
                "disease_id": predicted_class,
                "disease_name": disease_info["disease_name"],
                "plant": disease_info["plant"],
                "scientific_name": disease_info.get("scientific_name"),
                "confidence": confidence,
                "confidence_percentage": round(confidence * 100, 2),
                "is_healthy": is_healthy,
                "is_uncertain": is_uncertain,
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
                "model_id": selected_model_id,
                "model_name": model_config["name"],
                "model_architecture": model_config["architecture"],
                "model_accuracy": model_config.get("accuracy", "N/A"),
                "timestamp": datetime.now().isoformat(),
                "device": str(device)
            }
        }
        
        logger.info(f"Prediction [{selected_model_id}]: {predicted_class} ({confidence:.2%})")
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
    ║     🌿 Plant Disease Detection API v2.0 (Multi-Model) 🌿 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Endpoints:                                              ║
    ║  • GET  /           - API info                           ║
    ║  • GET  /health     - Health check                       ║
    ║  • GET  /models     - List all available models          ║
    ║  • GET  /models/<id> - Get model details                 ║
    ║  • GET  /diseases   - List all diseases                  ║
    ║  • GET  /disease/<id> - Disease details                  ║
    ║  • POST /predict    - Upload image for prediction        ║
    ║        (optional: ?model=<model_id> to select model)     ║
    ║  • GET  /stats      - API statistics                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"    📊 Models loaded: {len(loaded_models)}/{len(MODEL_REGISTRY)}")
    print(f"    🎯 Default model: {default_model_id}")
    print(f"    💻 Device: {device}")
    print(f"    🦠 Disease classes: {NUM_CLASSES}")
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
