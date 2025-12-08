from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'plant_disease_model.pth')

# Load class indices from JSON file
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), 'class_indices.json')
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices_dict = json.load(f)

# Convert to list for easy indexing
DISEASE_CLASSES = [class_indices_dict[str(i)] for i in range(len(class_indices_dict))]
NUM_CLASSES = len(DISEASE_CLASSES)

# Load the model
def load_model(model_path, num_classes):
    """Load PyTorch model from .pth file"""
    try:
        # Create a ResNet50 model (adjust architecture if your model uses different backbone)
        model = models.resnet50(weights=None)
        # Modify the final layer to match number of disease classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

try:
    model = load_model(MODEL_PATH, NUM_CLASSES)
    model_loaded = True
except Exception as e:
    print(f"Warning: {str(e)}")
    model = None
    model_loaded = False

# Define image preprocessing
def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

preprocess = get_transforms()

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

@app.route('/')
def home():
    return jsonify({"message": "Plant Disease Prediction API (PyTorch) is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_loaded:
            return jsonify({"error": "Model not loaded. Please check the model file."}), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Check if file is an image
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({"error": "Invalid file format. Please upload an image file."}), 400
        
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        image_tensor = preprocess_image(image)
        
        # Move to device
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
        predicted_disease = DISEASE_CLASSES[predicted_class_index] if predicted_class_index < len(DISEASE_CLASSES) else "Unknown"
        
        # Format the disease name for better readability
        formatted_disease = predicted_disease.replace('___', ' - ').replace('_', ' ')
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "disease": DISEASE_CLASSES[i].replace('___', ' - ').replace('_', ' '),
                "confidence": float(probabilities[i])
            }
            for i in top_indices
        ]
        
        return jsonify({
            "success": True,
            "predicted_disease": formatted_disease,
            "confidence": confidence,
            "top_predictions": top_predictions
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device),
        "num_classes": NUM_CLASSES
    })

if __name__ == '__main__':
    print("Starting Plant Disease Prediction API (PyTorch)...")
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Loaded {NUM_CLASSES} disease classes")
    print(f"Using device: {device}")
    print(f"Model status: {'Ready' if model_loaded else 'Failed to load'}")
    print()
    print("IMPORTANT FOR MOBILE CAMERA:")
    print("- Camera requires HTTPS on mobile browsers")
    print("- Start frontend with HTTPS using: npm run start:https")
    print("- Or run: start-https.bat in frontend folder")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
