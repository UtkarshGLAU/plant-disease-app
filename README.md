# 🌿 PlantGuard AI - Plant Disease Detection System

<div align="center">

![PlantGuard AI](https://img.shields.io/badge/PlantGuard-AI-10b981?style=for-the-badge&logo=leaf&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet34-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)

**An AI-Powered Plant Disease Detection System for Agricultural Health Monitoring**

*BTech Final Year Project*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Supported Diseases](#-supported-diseases)
- [Model Information](#-model-information)
- [Screenshots](#-screenshots)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🌟 Overview

PlantGuard AI is a comprehensive plant disease detection system that leverages deep learning to identify plant diseases from leaf images. The system provides instant diagnosis along with detailed treatment recommendations and prevention tips.

### Key Highlights

- 🎯 **38+ Disease Detection** - Accurately identify diseases across various crops
- ⚡ **Real-time Analysis** - Get results in seconds
- 💊 **Treatment Recommendations** - Detailed advice for each condition
- 📱 **Mobile Ready** - Responsive design with camera support
- 🔬 **Multi-Model Support** - Choose from different AI models (ResNet, VGG, EfficientNet, etc.)
- 🧠 **Model Selection** - Users can select which model to use for analysis

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Upload** | Drag-and-drop or click-to-upload interface |
| 📷 **Camera Capture** | Real-time camera integration for mobile devices |
| 🤖 **AI Analysis** | Multiple deep learning models available |
| 🧠 **Model Selection** | Choose the AI model that best fits your needs |
| 📊 **Confidence Scores** | Probability distribution for top predictions |
| 💡 **Smart Recommendations** | Disease-specific treatment and prevention tips |
| 🌡️ **Severity Assessment** | Visual severity indicators for each condition |

### User Experience

- ✅ Beautiful animated dark-themed UI
- ✅ Smooth transitions and micro-interactions
- ✅ Responsive design for all screen sizes
- ✅ Real-time backend health monitoring
- ✅ Comprehensive error handling

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework
- **PyTorch** - Deep learning framework
- **torchvision** - Image transformations
- **Pillow** - Image processing

### Frontend
- **React 18** - UI framework
- **CSS3** - Styling with animations
- **Modern JavaScript (ES6+)** - Application logic

### Model
- **ResNet34** - Pre-trained on ImageNet
- **Transfer Learning** - Fine-tuned on PlantVillage dataset
- **38 Classes** - Various plant diseases and healthy conditions

---

## 📁 Project Structure

```
G180/
├── 📁 backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── class_indices.json     # Disease class mappings
│   ├── disease_info.json      # Disease recommendations database
│   └── start-backend.bat      # Windows startup script
│
├── 📁 frontend/
│   ├── 📁 public/
│   │   ├── index.html         # HTML template
│   │   └── manifest.json      # PWA manifest
│   ├── 📁 src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Component styles
│   │   ├── index.js           # React entry point
│   │   └── index.css          # Global styles
│   ├── package.json           # Node.js dependencies
│   └── start-frontend.bat     # Windows startup script
│
├── 📁 model/
│   └── plantDisease-resnet34.pth  # Trained PyTorch model
│
├── setup.bat                  # Complete setup script
└── README.md                  # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- Git

### Quick Setup (Windows)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd G180
   ```

2. **Run the setup script**
   ```bash
   setup.bat
   ```

### Manual Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

---

## 💻 Usage

### Starting the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   start-backend.bat
   # Or: python app.py
   ```
   The API will be available at `http://localhost:5000`

2. **Start the Frontend Server**
   ```bash
   cd frontend
   start-frontend.bat
   # Or: npm start
   ```
   The application will open at `http://localhost:3000`

### Using the Application

1. **Upload Image**: Drag and drop or click to select a plant leaf image
2. **Or Use Camera**: Click the camera button to capture a live photo
3. **Analyze**: Click "Analyze Plant" to get AI-powered diagnosis
4. **View Results**: See disease prediction, confidence score, and recommendations

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "timestamp": "2026-01-31T10:00:00",
  "classes_available": 38
}
```

#### Predict Disease
```http
POST /predict
Content-Type: multipart/form-data
```
**Request Body:**
- `image`: Image file (PNG, JPG, JPEG, WEBP)

**Response:**
```json
{
  "success": true,
  "prediction": {
    "disease_id": "Tomato___Late_blight",
    "disease_name": "Late Blight",
    "plant": "Tomato",
    "confidence": 0.95,
    "confidence_percentage": 95.0,
    "is_healthy": false,
    "severity": "critical",
    "severity_color": "#ef4444"
  },
  "disease_info": {
    "description": "Late blight is...",
    "symptoms": ["..."],
    "causes": ["..."],
    "treatment": ["..."],
    "prevention": ["..."],
    "is_contagious": true
  },
  "top_predictions": [...]
}
```

#### List All Diseases
```http
GET /diseases
```

#### Get Disease Details
```http
GET /disease/<disease_id>
```

#### Get Statistics
```http
GET /stats
```

---

## 🦠 Supported Diseases

### Plants & Conditions (38 Classes)

| Plant | Diseases | Healthy |
|-------|----------|---------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust | ✓ |
| 🫐 Blueberry | - | ✓ |
| 🍒 Cherry | Powdery Mildew | ✓ |
| 🌽 Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight | ✓ |
| 🍇 Grape | Black Rot, Esca, Leaf Blight | ✓ |
| 🍊 Orange | Huanglongbing (Citrus Greening) | - |
| 🍑 Peach | Bacterial Spot | ✓ |
| 🫑 Pepper | Bacterial Spot | ✓ |
| 🥔 Potato | Early Blight, Late Blight | ✓ |
| 🫐 Raspberry | - | ✓ |
| 🫘 Soybean | - | ✓ |
| 🎃 Squash | Powdery Mildew | - |
| 🍓 Strawberry | Leaf Scorch | ✓ |
| 🍅 Tomato | 9 Diseases (Bacterial Spot, Early/Late Blight, Leaf Mold, etc.) | ✓ |

---

## 🧠 Model Information

### Multi-Model Support

PlantGuard AI supports multiple deep learning models. Users can select their preferred model from the dropdown before analysis.

#### Supported Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| **ResNet34** | Default model, balanced performance | General use |
| **ResNet50** | Deeper network, higher accuracy | Complex cases |
| **EfficientNet-B0** | Efficient and accurate | Resource-limited devices |
| **VGG16** | Classic architecture | Comparison baseline |
| **MobileNet** | Lightweight model | Mobile devices |

### Adding New Models

To add a new model, update the `MODEL_REGISTRY` in `backend/app.py`:

```python
MODEL_REGISTRY = {
    "your-model-id": {
        "name": "Your Model Name",
        "path": "model/your-model.pth",
        "architecture": "resnet34",  # or resnet50, efficientnet, vgg16, etc.
        "num_classes": 38,
        "accuracy": 0.94,
        "is_default": False
    }
}
```

### Default Architecture (ResNet34)
- **Base Model**: ResNet34 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned on PlantVillage dataset
- **Input Size**: 224 × 224 pixels
- **Output**: 38 classes (softmax probabilities)

### Preprocessing
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Performance
- Training Accuracy: ~97%
- Validation Accuracy: ~95%
- Test Accuracy: ~94%

---

## 🖼️ Screenshots

### Home Screen
- Beautiful dark-themed interface
- Animated background with floating particles
- Real-time API status indicator

### Analysis Results
- Disease identification with confidence score
- Tabbed interface for overview, treatment, and predictions
- Severity indicators and contagion warnings

### Mobile View
- Responsive design
- Camera integration
- Touch-friendly interface

---

## 🔧 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Model Not Loading
- Ensure `plantDisease-resnet34.pth` exists in the `model/` directory
- Check file isn't corrupted (should be ~85MB)
- Verify PyTorch version compatibility

#### Frontend Connection Failed
- Ensure backend is running on port 5000
- Check firewall settings
- Verify CORS is enabled

#### Camera Not Working
- Use HTTPS for production (required by browsers)
- Grant camera permissions
- Try different browser

### Windows Firewall Rules
```powershell
# Allow Flask backend
netsh advfirewall firewall add rule name="PlantGuard Backend" dir=in action=allow protocol=TCP localport=5000

# Allow React frontend
netsh advfirewall firewall add rule name="PlantGuard Frontend" dir=in action=allow protocol=TCP localport=3000
```

---

## 🚀 Future Enhancements

- [ ] MongoDB integration for disease database
- [ ] User authentication and history
- [ ] Multi-language support
- [ ] Offline PWA capabilities
- [ ] Weather-based recommendations
- [ ] Batch image processing
- [ ] Export reports as PDF
- [ ] Integration with agricultural APIs

---

## 📄 License

This project is created for educational and research purposes as part of BTech Final Year Project.

---

## 🙏 Acknowledgments

- PlantVillage Dataset
- PyTorch Team
- React Community
- All contributors and testers

---

<div align="center">

**Made with 💚 for Agricultural Health**

🌿 PlantGuard AI - Protecting Plants, One Leaf at a Time 🌿

</div>
