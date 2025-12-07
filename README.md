# Plant Disease Detection System

A full-stack application for detecting plant diseases using machine learning. The system consists of a React frontend and Flask backend that uses your trained TensorFlow model.

## Project Structure

```
G180/
├── model/
│   └── plant_disease_prediction_model.h5
├── backend/
│   ├── app.py
│   └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js
│       ├── App.css
│       ├── index.js
│       └── index.css
└── README.md
```

## Features

- 🖼️ **Image Upload**: Easy drag-and-drop or click-to-upload interface
- 🤖 **AI Analysis**: Uses your trained TensorFlow model for disease prediction
- 📊 **Detailed Results**: Shows primary diagnosis with confidence scores
- 📋 **Top Predictions**: Displays top 3 most likely diseases
- 💡 **Recommendations**: Provides actionable advice based on results
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### Backend Setup

1. **Navigate to the backend directory:**
   ```powershell
   cd backend
   ```

2. **Create a Python virtual environment:**
   ```powershell
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   ```powershell
   venv\Scripts\Activate.ps1
   ```

4. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Start the Flask server:**
   ```powershell
   python app.py
   ```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Open a new terminal and navigate to the frontend directory:**
   ```powershell
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```powershell
   npm install
   ```

3. **Start the React development server:**
   
   **For HTTP (desktop only):**
   ```powershell
   npm start
   ```
   
   **For HTTPS (required for mobile camera):**
   ```powershell
   npm run start:https
   ```
   
   **Or use the batch file (Windows):**
   ```powershell
   .\start-https.bat
   ```

**Important for Mobile Camera Access:**
- 📱 Mobile browsers (especially Firefox) require HTTPS for camera access
- 🔒 Use HTTPS mode when testing on mobile devices
- ⚠️ You'll see a security warning for the self-signed certificate (this is normal for development)
- ✅ Click "Advanced" → "Proceed to [address] (unsafe)" in your browser

The frontend will start on:
- HTTP: `http://localhost:3000`
- HTTPS: `https://localhost:3000`

## Usage

1. Open your browser and go to `http://localhost:3000`
2. Click "Choose Image File" to select a plant leaf image
3. Preview your selected image
4. Click "Analyze Plant" to get disease predictions
5. View the results including:
   - Primary diagnosis with confidence score
   - Health status (Healthy/Diseased)
   - Top 3 predictions with confidence bars
   - Recommendations based on the diagnosis

## API Endpoints

### Backend API (`http://localhost:5000`)

- **GET `/`** - Health check endpoint
- **POST `/predict`** - Upload image and get disease prediction
  - Accept: `multipart/form-data`
  - Field: `image` (image file)
  - Returns: JSON with prediction results
- **GET `/health`** - Check if model is loaded properly

## Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP

## Model Information

The system uses your pre-trained TensorFlow model (`plant_disease_prediction_model.h5`) which should be capable of detecting various plant diseases. The model expects 224x224 pixel RGB images.

### Supported Disease Classes

The system currently supports detection of 38 different plant conditions including:
- Apple diseases (scab, black rot, cedar apple rust)
- Corn diseases (leaf spot, rust, blight)
- Tomato diseases (bacterial spot, early blight, late blight, etc.)
- Grape, peach, pepper, potato, and other plant diseases
- Healthy plant detection

## Troubleshooting

### Common Issues

1. **Backend not starting:**
   - Ensure Python virtual environment is activated
   - Check if all dependencies are installed
   - Verify the model file path is correct

2. **Frontend not connecting to backend:**
   - Ensure backend is running on port 5000
   - Check CORS settings in Flask app
   - Verify frontend is making requests to correct URL

3. **Image upload fails:**
   - Check image format is supported
   - Ensure image size is reasonable (< 10MB recommended)
   - Verify backend has proper file handling permissions

4. **Model prediction errors:**
   - Ensure the model file is not corrupted
   - Check if model expects different input dimensions
   - Verify TensorFlow version compatibility

### Customization

To customize the disease classes, edit the `DISEASE_CLASSES` list in `backend/app.py` to match your model's output classes.

To modify the image preprocessing, update the `preprocess_image()` function in `backend/app.py` to match your model's expected input format.

## Development

### Adding New Features

1. **Backend**: Modify `app.py` to add new endpoints or functionality
2. **Frontend**: Update React components in the `src/` directory
3. **Styling**: Modify `App.css` for visual changes

### Testing

- Backend: Test API endpoints using tools like Postman
- Frontend: Use React Developer Tools for component debugging

## Production Deployment

For production deployment, consider:

1. **Backend**: Use production WSGI server like Gunicorn
2. **Frontend**: Build the React app with `npm run build`
3. **Environment**: Set appropriate environment variables
4. **Security**: Enable HTTPS and proper authentication

## License

This project is created for educational and research purposes.

---

## 📱 **Mobile Testing & Troubleshooting**

### **Testing URLs:**
- **Main App**: `http://172.16.144.249:3000`
- **Camera Test**: `http://172.16.144.249:3000/camera-test.html`
- **Backend Test**: `http://172.16.144.249:3000/backend-test.html`

### **Common Mobile Issues:**

#### **1. Camera Not Working:**
- **Firefox Mobile**: Requires HTTPS - use ngrok or cloudflare tunnel
- **Chrome/Edge Mobile**: Usually works with HTTP
- **Safari Mobile**: May require HTTPS
- **Solution**: Try different browsers or set up HTTPS tunnel

#### **2. Prediction Fails on Mobile:**
- **Backend Connection**: Check if backend is accessible from mobile
- **Network Issues**: Ensure devices are on same WiFi network
- **Firewall**: Windows Firewall may block port 5000
- **Solution**: Test backend connectivity first

#### **3. Backend Connection Issues:**

**Check Backend Status:**
1. Visit: `http://172.16.144.249:3000/backend-test.html`
2. Click "Test Backend Connection"
3. If fails, check the issues below

**Common Causes:**
- Backend server not running
- Wrong IP address (check with `ipconfig`)
- Windows Firewall blocking port 5000
- Antivirus software blocking connections
- Mobile and PC on different networks

**Solutions:**
1. **Windows Firewall**: Add exceptions for ports 3000 and 5000
2. **Check Backend**: Ensure Flask server is running on `0.0.0.0:5000`
3. **Network**: Ensure both devices on same WiFi
4. **IP Address**: Update IP in code if computer IP changed

### **Quick Mobile Setup:**

1. **Start Backend:**
   ```bash
   cd backend
   python app.py
   ```

2. **Start Frontend:**
   ```bash
   cd frontend
   npm start
   ```

3. **Test on Mobile:**
   - Connect phone to same WiFi
   - Visit: `http://172.16.144.249:3000/backend-test.html`
   - Verify backend connection works
   - Then visit: `http://172.16.144.249:3000`

4. **For Camera (Firefox Mobile):**
   - Install ngrok: https://ngrok.com/download
   - Run: `ngrok http 3000`
   - Use the https://xxx.ngrok-free.app URL

### **Windows Firewall Fix:**
If backend connection fails, add firewall rules:
```powershell
# Allow port 5000 (Flask backend)
netsh advfirewall firewall add rule name="Flask Backend" dir=in action=allow protocol=TCP localport=5000

# Allow port 3000 (React frontend)  
netsh advfirewall firewall add rule name="React Frontend" dir=in action=allow protocol=TCP localport=3000
```