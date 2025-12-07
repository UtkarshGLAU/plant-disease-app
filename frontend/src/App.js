import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const [videoReady, setVideoReady] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setPrediction(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Check backend health
  const checkBackendHealth = async () => {
    const currentHost = window.location.hostname;
    const protocol = window.location.protocol;
    
    const backendUrl = (currentHost !== 'localhost' && currentHost !== '127.0.0.1') 
      ? `${protocol}//${currentHost}:5000` 
      : 'http://localhost:5000';

    try {
      const response = await fetch(`${backendUrl}/health`, { 
        method: 'GET',
        timeout: 5000 
      });
      
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(data.model_loaded ? 'ready' : 'loading');
      } else {
        setBackendStatus('error');
      }
    } catch (err) {
      console.warn('Backend health check failed:', err);
      setBackendStatus('error');
    }
  };

  // Check backend health on component mount
  React.useEffect(() => {
    checkBackendHealth();
    // Check every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Check if camera is supported
  const isCameraSupported = () => {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  };

  // Check if on mobile device
  const isMobileDevice = () => {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  };

  const startCamera = async () => {
    try {
      setError(null);
      
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera not supported on this device or browser. Please use file upload instead.');
        return;
      }

      // Check if we're on HTTP (not HTTPS) which blocks camera on many browsers
      if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        setError('Camera requires HTTPS connection. Please use file upload or access via HTTPS.');
        return;
      }

      // First try with environment (back) camera preference
      let constraints = {
        video: {
          facingMode: { ideal: 'environment' }, // Prefer back camera
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      let mediaStream;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (envError) {
        // Fallback to any available camera
        console.warn('Environment camera not available, trying any camera:', envError);
        constraints = {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        };
        
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (anyError) {
          // Final fallback with minimal constraints
          console.warn('Standard camera constraints failed, using minimal:', anyError);
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        }
      }

      setStream(mediaStream);
      setShowCamera(true);
      setVideoReady(false);
      
      // Wait for video element to be ready and set up the stream
      setTimeout(() => {
        if (videoRef.current && mediaStream) {
          videoRef.current.srcObject = mediaStream;
          
          // Add event listeners for video
          videoRef.current.onloadedmetadata = () => {
            setVideoReady(true);
            if (videoRef.current) {
              videoRef.current.play().catch(playError => {
                console.warn('Auto-play failed, user interaction needed:', playError);
                // Video will show play button if autoplay fails
              });
            }
          };

          // Handle play/pause events
          videoRef.current.onplay = () => setVideoReady(true);
          videoRef.current.onpause = () => setVideoReady(false);
        }
      }, 100);

    } catch (err) {
      console.error('Error accessing camera:', err);
      let errorMessage = 'Unable to access camera. ';
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera permissions and try again.';
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device.';
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Camera not supported on this browser.';
      } else if (err.name === 'NotReadableError') {
        errorMessage += 'Camera is being used by another application.';
      } else {
        errorMessage += 'Please check camera permissions and try again.';
      }
      
      setError(errorMessage);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setShowCamera(false);
    setVideoReady(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert canvas to blob and create file
      canvas.toBlob((blob) => {
        const capturedFile = new File([blob], 'captured-plant-image.jpg', { type: 'image/jpeg' });
        setSelectedFile(capturedFile);
        
        // Create preview from canvas
        const dataURL = canvas.toDataURL('image/jpeg');
        setPreview(dataURL);
        
        // Stop camera and hide camera view
        stopCamera();
        setPrediction(null);
      }, 'image/jpeg', 0.8);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Determine the backend URL based on current environment
    const getBackendUrl = () => {
      const currentHost = window.location.hostname;
      const protocol = window.location.protocol;
      
      // If accessing from mobile (not localhost), use the computer's IP
      if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
        // Use the same host but port 5000 for Flask backend
        return `${protocol}//${currentHost}:5000`;
      }
      
      // Default to localhost for local development
      return 'http://localhost:5000';
    };

    const backendUrl = getBackendUrl();

    try {
      const response = await axios.post(`${backendUrl}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout for mobile
      });

      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      let errorMessage = 'An error occurred during prediction. ';
      
      if (err.code === 'NETWORK_ERROR' || err.code === 'ERR_NETWORK') {
        errorMessage += `Cannot connect to backend server at ${backendUrl}. Please make sure the backend is running.`;
      } else if (err.code === 'TIMEOUT' || err.code === 'ECONNABORTED') {
        errorMessage += 'Request timed out. Please try again.';
      } else if (err.response?.status === 404) {
        errorMessage += `Backend server not found at ${backendUrl}. Please check the backend is running.`;
      } else if (err.response?.status >= 500) {
        errorMessage += 'Server error occurred. Please try again.';
      } else {
        errorMessage += err.response?.data?.error || err.message || 'Please try again.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    stopCamera(); // Stop camera if it's running
    document.getElementById('file-input').value = '';
  };

  // Cleanup camera stream when component unmounts
  React.useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const getHealthStatus = (diseaseName) => {
    return diseaseName.toLowerCase().includes('healthy') ? 'healthy' : 'diseased';
  };

  const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(2);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🌱 Plant Disease Detection</h1>
        <p>Upload an image of a plant leaf to detect diseases</p>
        
        <div className={`backend-status ${backendStatus}`}>
          {backendStatus === 'checking' && '🔄 Connecting to server...'}
          {backendStatus === 'ready' && '✅ Server ready'}
          {backendStatus === 'loading' && '⏳ Server loading model...'}
          {backendStatus === 'error' && '❌ Server unavailable'}
        </div>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="input-methods">
            <h3>Choose Input Method:</h3>
            
            {window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1' && (
              <div className="https-warning">
                <p>⚠️ Camera requires HTTPS connection. For camera access, please use file upload or access via HTTPS.</p>
              </div>
            )}
            
            <div className="method-buttons">
              <div className="file-input-container">
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
                <label htmlFor="file-input" className="file-input-label">
                  📁 Upload Image
                </label>
              </div>

              {isCameraSupported() && (
                <button
                  onClick={startCamera}
                  disabled={loading || showCamera}
                  className="camera-btn"
                >
                  📷 Use Camera
                </button>
              )}

              {!isCameraSupported() && (
                <div className="camera-not-supported">
                  <p>📷 Camera not available on this device</p>
                </div>
              )}
            </div>
          </div>

          {showCamera && (
            <div className="camera-section">
              <h3>📸 Camera View</h3>
              <div className="camera-container">
                <video
                  ref={videoRef}
                  className="camera-video"
                  autoPlay
                  playsInline
                  muted
                  controls={false}
                  webkit-playsinline="true"
                />
                <canvas
                  ref={canvasRef}
                  style={{ display: 'none' }}
                />
              </div>
              <div className="camera-controls">
                <button
                  onClick={capturePhoto}
                  className="capture-btn"
                  disabled={loading || !videoReady}
                >
                  📸 Capture Photo
                </button>
                <button
                  onClick={stopCamera}
                  className="stop-camera-btn"
                  disabled={loading}
                >
                  ❌ Cancel
                </button>
                {!videoReady && (
                  <div className="camera-status">
                    <p>📹 Starting camera...</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {selectedFile && !showCamera && (
            <div className="file-info">
              <p>Selected: {selectedFile.name}</p>
            </div>
          )}

          {preview && !showCamera && (
            <div className="preview-section">
              <h3>Image Preview:</h3>
              <img src={preview} alt="Preview" className="image-preview" />
            </div>
          )}

          {!showCamera && (
            <div className="button-group">
              <button
                onClick={handleUpload}
                disabled={!selectedFile || loading}
                className="upload-btn"
              >
                {loading ? '🔄 Analyzing...' : '🔍 Analyze Plant'}
              </button>
              
              <button
                onClick={resetForm}
                className="reset-btn"
                disabled={loading}
              >
                🗑️ Reset
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {prediction && prediction.success && (
          <div className="results-section">
            <h3>🎯 Analysis Results</h3>
            
            <div className={`main-prediction ${getHealthStatus(prediction.predicted_disease)}`}>
              <h4>Primary Diagnosis:</h4>
              <p className="disease-name">{prediction.predicted_disease}</p>
              <p className="confidence">
                Confidence: {formatConfidence(prediction.confidence)}%
              </p>
              
              <div className={`health-status ${getHealthStatus(prediction.predicted_disease)}`}>
                {getHealthStatus(prediction.predicted_disease) === 'healthy' ? '✅ Healthy' : '⚠️ Disease Detected'}
              </div>
            </div>

            {prediction.top_predictions && (
              <div className="top-predictions">
                <h4>Top 3 Possibilities:</h4>
                <div className="predictions-list">
                  {prediction.top_predictions.map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <div className="rank">#{index + 1}</div>
                      <div className="prediction-details">
                        <div className="disease">{pred.disease}</div>
                        <div className="confidence">{formatConfidence(pred.confidence)}%</div>
                      </div>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill" 
                          style={{ width: `${pred.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="recommendations">
              <h4>💡 Recommendations:</h4>
              {getHealthStatus(prediction.predicted_disease) === 'healthy' ? (
                <ul>
                  <li>Your plant appears to be healthy! 🎉</li>
                  <li>Continue with regular care and monitoring</li>
                  <li>Maintain proper watering and lighting conditions</li>
                </ul>
              ) : (
                <ul>
                  <li>Consider consulting with a plant pathologist for treatment options</li>
                  <li>Isolate the affected plant to prevent spread</li>
                  <li>Remove affected leaves if recommended for the specific disease</li>
                  <li>Adjust watering and environmental conditions as needed</li>
                </ul>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Plant Disease Detection System - Powered by AI 🤖</p>
      </footer>
    </div>
  );
}

export default App;