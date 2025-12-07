from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'plant_disease_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices from JSON file
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), 'class_indices.json')
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices_dict = json.load(f)

# Convert to list for easy indexing
DISEASE_CLASSES = [class_indices_dict[str(i)] for i in range(len(class_indices_dict))]

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Resize image to match model input size (assuming 224x224, adjust if different)
    image = image.resize((224, 224))
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def home():
    return jsonify({"message": "Plant Disease Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Get predicted disease class
        predicted_disease = DISEASE_CLASSES[predicted_class_index] if predicted_class_index < len(DISEASE_CLASSES) else "Unknown"
        
        # Format the disease name for better readability
        formatted_disease = predicted_disease.replace('___', ' - ').replace('_', ' ')
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_predictions = [
            {
                "disease": DISEASE_CLASSES[i].replace('___', ' - ').replace('_', ' '),
                "confidence": float(predictions[0][i])
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
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    print("Starting Plant Disease Prediction API...")
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Loaded {len(DISEASE_CLASSES)} disease classes")
    print()
    print("IMPORTANT FOR MOBILE CAMERA:")
    print("- Camera requires HTTPS on mobile browsers")
    print("- Start frontend with HTTPS using: npm run start:https")
    print("- Or run: start-https.bat in frontend folder")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)