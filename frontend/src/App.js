import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// API Configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Floating Particles Component
const Particles = () => {
  const particles = Array.from({ length: 20 }, (_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    animationDelay: `${Math.random() * 15}s`,
    animationDuration: `${15 + Math.random() * 10}s`,
  }));

  return (
    <div className="particles">
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="particle"
          style={{
            left: particle.left,
            animationDelay: particle.animationDelay,
            animationDuration: particle.animationDuration,
          }}
        />
      ))}
    </div>
  );
};

// Loading Overlay Component
const LoadingOverlay = ({ message }) => (
  <div className="loading-overlay">
    <div className="loading-spinner" />
    <p className="loading-text">{message || 'Analyzing your plant...'}</p>
  </div>
);

// Header Component
const Header = ({ backendStatus, models, selectedModel, onModelChange }) => (
  <header className="header">
    <div className="header-content">
      <div className="logo">
        <div className="logo-icon">🌿</div>
        <div>
          <span className="logo-text">PlantGuard</span>
          <span className="logo-badge">AI</span>
        </div>
      </div>
      <div className="header-right">
        {/* Model Selector in Header */}
        <div className="header-model-selector">
          <span className="model-icon">🧠</span>
          <select
            className="header-model-dropdown"
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={models.length === 0}
          >
            {models.length === 0 ? (
              <option value="">Loading...</option>
            ) : (
              models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} {model.is_default ? '★' : ''}
                </option>
              ))
            )}
          </select>
        </div>
        <div className="status-badge">
          <span
            className={`status-dot ${
              backendStatus === 'ready'
                ? 'online'
                : backendStatus === 'error'
                ? 'offline'
                : 'loading'
            }`}
          />
          <span>
            {backendStatus === 'ready'
              ? 'Ready'
              : backendStatus === 'error'
              ? 'Offline'
              : 'Connecting...'}
          </span>
        </div>
      </div>
    </div>
  </header>
);

// Hero Section Component
const Hero = () => (
  <section className="hero">
    <h1 className="hero-title">
      Detect Plant Diseases with <span className="gradient-text">AI Precision</span>
    </h1>
    <p className="hero-subtitle">
      Upload or capture a photo of your plant leaf and get instant disease diagnosis 
      with treatment recommendations powered by advanced machine learning.
    </p>
  </section>
);

// Upload Card Component
const UploadCard = ({
  preview,
  isDragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileSelect,
  onRemove,
  onCameraStart,
  fileInputRef,
  fileName,
}) => (
  <div className="card">
    <div className="card-header">
      <div className="card-icon">📤</div>
      <div>
        <h2 className="card-title">Upload Plant Image</h2>
        <p className="card-subtitle">Drag & drop or click to upload</p>
      </div>
    </div>

    {!preview ? (
      <>
        <div
          className={`upload-zone ${isDragging ? 'dragover' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">🌱</div>
          <p className="upload-text">Drop your plant image here</p>
          <p className="upload-hint">or click to browse files</p>
          <div className="upload-formats">
            <span className="format-badge">PNG</span>
            <span className="format-badge">JPG</span>
            <span className="format-badge">JPEG</span>
            <span className="format-badge">WEBP</span>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={onFileSelect}
          className="hidden-input"
        />

        <div className="action-buttons">
          <button
            className="btn btn-secondary"
            onClick={() => fileInputRef.current?.click()}
          >
            📁 Choose File
          </button>
          <button className="btn btn-camera" onClick={onCameraStart}>
            📷 Use Camera
          </button>
        </div>
      </>
    ) : (
      <div className="preview-container">
        <img src={preview} alt="Preview" className="preview-image" />
        <div className="preview-overlay">
          <span className="preview-info">{fileName || 'Captured Image'}</span>
          <button className="remove-btn" onClick={onRemove}>
            ✕ Remove
          </button>
        </div>
      </div>
    )}
  </div>
);

// Camera Modal Component
const CameraModal = ({ videoRef, onCapture, onClose, videoReady }) => (
  <div className="card">
    <div className="card-header">
      <div className="card-icon">📷</div>
      <div>
        <h2 className="card-title">Camera Capture</h2>
        <p className="card-subtitle">Position your plant leaf in frame</p>
      </div>
    </div>

    <div className="camera-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-video"
      />
      <div className="camera-overlay">
        <div className="camera-controls">
          <button
            className="btn btn-primary"
            onClick={onCapture}
            disabled={!videoReady}
          >
            📸 Capture Photo
          </button>
          <button className="btn btn-secondary" onClick={onClose}>
            ✕ Cancel
          </button>
        </div>
      </div>
    </div>
  </div>
);

// Results Card Component
const ResultsCard = ({ result, activeTab, setActiveTab }) => {
  if (!result) return null;

  const { prediction, disease_info, top_predictions, metadata, is_uncertain, uncertainty_message } = result;

  return (
    <div className="card results-section">
      {/* Uncertainty Warning */}
      {is_uncertain && (
        <div className="uncertainty-warning">
          <span className="uncertainty-icon">⚠️</span>
          <div className="uncertainty-content">
            <strong>Low Confidence Result</strong>
            <p>{uncertainty_message || "This plant may not be in our training data. Results may be inaccurate."}</p>
          </div>
        </div>
      )}

      <div className="card-header">
        <div className="card-icon">🔬</div>
        <div>
          <h2 className="card-title">Analysis Results</h2>
          <p className="card-subtitle">
            AI-powered diagnosis
            {metadata?.model_name && (
              <span className="model-badge">🧠 {metadata.model_name}</span>
            )}
          </p>
        </div>
      </div>

      {/* Result Header */}
      <div className="result-header">
        <div className={`result-icon ${prediction.is_healthy ? 'healthy' : 'diseased'}`}>
          {prediction.is_healthy ? '✓' : '⚠'}
        </div>
        <div className="result-info">
          <h3>{prediction.disease_name}</h3>
          <p className="result-plant">{prediction.plant}</p>
          {prediction.scientific_name && (
            <p className="result-scientific">{prediction.scientific_name}</p>
          )}
        </div>
        <div
          className="severity-badge"
          style={{
            background: `${prediction.severity_color}20`,
            color: prediction.severity_color,
            border: `1px solid ${prediction.severity_color}40`,
          }}
        >
          {prediction.is_healthy ? '🌟 Healthy' : `⚡ ${prediction.severity}`}
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="confidence-section">
        <div className="confidence-header">
          <span className="confidence-label">Confidence Level</span>
          <span className="confidence-value">{prediction.confidence_percentage}%</span>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${prediction.confidence_percentage}%` }}
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button
          className={`tab ${activeTab === 'treatment' ? 'active' : ''}`}
          onClick={() => setActiveTab('treatment')}
        >
          Treatment
        </button>
        <button
          className={`tab ${activeTab === 'predictions' ? 'active' : ''}`}
          onClick={() => setActiveTab('predictions')}
        >
          All Results
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="info-grid">
          <div className="info-card">
            <div className="info-card-header">
              <span className="info-card-icon">📋</span>
              <span className="info-card-title">Description</span>
            </div>
            <p className="info-card-content">{disease_info.description}</p>
          </div>

          {disease_info.symptoms && disease_info.symptoms.length > 0 && (
            <div className="info-card symptoms">
              <div className="info-card-header">
                <span className="info-card-icon">🔍</span>
                <span className="info-card-title">Symptoms</span>
              </div>
              <ul className="info-card-list">
                {disease_info.symptoms.map((symptom, idx) => (
                  <li key={idx}>{symptom}</li>
                ))}
              </ul>
            </div>
          )}

          {disease_info.causes && disease_info.causes.length > 0 && (
            <div className="info-card causes">
              <div className="info-card-header">
                <span className="info-card-icon">⚠️</span>
                <span className="info-card-title">Causes</span>
              </div>
              <ul className="info-card-list">
                {disease_info.causes.map((cause, idx) => (
                  <li key={idx}>{cause}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {activeTab === 'treatment' && (
        <div className="info-grid">
          {disease_info.treatment && disease_info.treatment.length > 0 && (
            <div className="info-card treatment">
              <div className="info-card-header">
                <span className="info-card-icon">💊</span>
                <span className="info-card-title">Treatment Recommendations</span>
              </div>
              <ul className="info-card-list">
                {disease_info.treatment.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {disease_info.prevention && disease_info.prevention.length > 0 && (
            <div className="info-card prevention">
              <div className="info-card-header">
                <span className="info-card-icon">🛡️</span>
                <span className="info-card-title">Prevention Tips</span>
              </div>
              <ul className="info-card-list">
                {disease_info.prevention.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {disease_info.is_contagious && (
            <div className="info-card causes">
              <div className="info-card-header">
                <span className="info-card-icon">🦠</span>
                <span className="info-card-title">Contagion Warning</span>
              </div>
              <p className="info-card-content">
                This condition is contagious and can spread to other plants. 
                Isolate affected plants and sanitize tools after use.
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'predictions' && (
        <div className="top-predictions">
          <h4 className="predictions-title">Top 5 Predictions</h4>
          {top_predictions.map((pred, idx) => (
            <div key={idx} className="prediction-item">
              <div className="prediction-rank">{idx + 1}</div>
              <div className="prediction-info">
                <p className="prediction-name">{pred.disease_name}</p>
                <p className="prediction-plant">{pred.plant}</p>
              </div>
              <span className="prediction-confidence">
                {pred.confidence_percentage}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Features Section Component
const FeaturesSection = () => (
  <section className="features-section">
    <h2 className="features-title">Why Choose PlantGuard AI?</h2>
    <div className="features-grid">
      <div className="feature-card">
        <div className="feature-icon">🎯</div>
        <h3 className="feature-title">38+ Disease Detection</h3>
        <p className="feature-description">
          Accurately identify 38 different plant conditions across various crops
        </p>
      </div>
      <div className="feature-card">
        <div className="feature-icon">⚡</div>
        <h3 className="feature-title">Instant Analysis</h3>
        <p className="feature-description">
          Get results in seconds with our optimized deep learning model
        </p>
      </div>
      <div className="feature-card">
        <div className="feature-icon">💊</div>
        <h3 className="feature-title">Treatment Advice</h3>
        <p className="feature-description">
          Receive detailed treatment and prevention recommendations
        </p>
      </div>
      <div className="feature-card">
        <div className="feature-icon">📱</div>
        <h3 className="feature-title">Mobile Friendly</h3>
        <p className="feature-description">
          Use your phone camera to capture and analyze plants on-the-go
        </p>
      </div>
    </div>
  </section>
);

// Footer Component
const Footer = () => (
  <footer className="footer">
    <p className="footer-text">
      🌿 PlantGuard AI - BTech Final Year Project © {new Date().getFullYear()}
      <br />
      Powered by PyTorch ResNet34 • Built with React
    </p>
  </footer>
);

// Model Selector Component
const ModelSelector = ({ models, selectedModel, onModelChange, disabled }) => (
  <div className="model-selector">
    <div className="model-selector-header">
      <span className="model-icon">🧠</span>
      <span className="model-label">AI Model</span>
    </div>
    <div className="model-dropdown-wrapper">
      <select
        className="model-dropdown"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={disabled || models.length === 0}
      >
        {models.length === 0 ? (
          <option value="">Loading models...</option>
        ) : (
          models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} {model.is_default ? '(Default)' : ''} - {model.accuracy} accuracy
            </option>
          ))
        )}
      </select>
      <span className="dropdown-arrow">▼</span>
    </div>
    {selectedModel && models.length > 0 && (
      <div className="model-info">
        {(() => {
          const model = models.find(m => m.id === selectedModel);
          return model ? (
            <>
              <span className="model-architecture">{model.architecture}</span>
              <span className="model-dot">•</span>
              <span className="model-classes">{model.num_classes} classes</span>
            </>
          ) : null;
        })()}
      </div>
    )}
  </div>
);

// Main App Component
function App() {
  // State
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);
  const [videoReady, setVideoReady] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  // Refs
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/health`, {
        method: 'GET',
      });

      if (response.ok) {
        const data = await response.json();
        setBackendStatus(data.model_loaded ? 'ready' : 'loading');
      } else {
        setBackendStatus('error');
      }
    } catch (err) {
      console.error('Backend health check failed:', err);
      setBackendStatus('error');
    }
  }, []);

  // Fetch available models
  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/models`, {
        method: 'GET',
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.models) {
          setModels(data.models);
          // Set default model
          const defaultModel = data.models.find(m => m.is_default);
          if (defaultModel) {
            setSelectedModel(defaultModel.id);
          } else if (data.models.length > 0) {
            setSelectedModel(data.models[0].id);
          }
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }, []);

  useEffect(() => {
    checkBackendHealth();
    fetchModels();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, [checkBackendHealth, fetchModels]);

  // File handling
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    setSelectedFile(file);
    setFileName(file.name);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setPreview(null);
    setFileName('');
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  };

  // Camera handling
  const startCamera = async () => {
    try {
      setError(null);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera not supported on this device');
        return;
      }

      const constraints = {
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      let mediaStream;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (envError) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      setStream(mediaStream);
      setShowCamera(true);
      setVideoReady(false);

      setTimeout(() => {
        if (videoRef.current && mediaStream) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.onloadedmetadata = () => {
            setVideoReady(true);
            videoRef.current?.play().catch(console.warn);
          };
        }
      }, 100);
    } catch (err) {
      console.error('Camera error:', err);
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
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

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(
        (blob) => {
          const capturedFile = new File([blob], 'captured-plant-image.jpg', {
            type: 'image/jpeg',
          });
          setSelectedFile(capturedFile);
          setFileName('Camera Capture');
          setPreview(canvas.toDataURL('image/jpeg'));
          stopCamera();
          setResult(null);
          setError(null);
        },
        'image/jpeg',
        0.9
      );
    }
  };

  // Analyze image
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    if (backendStatus !== 'ready') {
      setError('AI model is not ready. Please wait and try again.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    if (selectedModel) {
      formData.append('model', selectedModel);
    }

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        setActiveTab('overview');
      } else {
        setError(data.error || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Failed to connect to the server. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="animated-background" />
      <Particles />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {loading && <LoadingOverlay message="Analyzing your plant with AI..." />}

      <Header 
        backendStatus={backendStatus}
        models={models}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />

      <main className="main-content">
        <Hero />

        <div className="upload-section">
          {/* Left Column - Upload/Camera */}
          <div>
            {showCamera ? (
              <CameraModal
                videoRef={videoRef}
                onCapture={capturePhoto}
                onClose={stopCamera}
                videoReady={videoReady}
              />
            ) : (
              <UploadCard
                preview={preview}
                isDragging={isDragging}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onFileSelect={handleFileSelect}
                onRemove={handleRemove}
                onCameraStart={startCamera}
                fileInputRef={fileInputRef}
                fileName={fileName}
              />
            )}

            {/* Analyze Button */}
            {preview && !showCamera && (
              <button
                className="btn btn-analyze"
                onClick={handleAnalyze}
                disabled={loading || backendStatus !== 'ready'}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    🔬 Analyze Plant
                  </>
                )}
              </button>
            )}

            {/* Error Message */}
            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div>
            {result ? (
              <ResultsCard
                result={result}
                activeTab={activeTab}
                setActiveTab={setActiveTab}
              />
            ) : (
              <div className="card" style={{ opacity: 0.6 }}>
                <div className="card-header">
                  <div className="card-icon">🔬</div>
                  <div>
                    <h2 className="card-title">Analysis Results</h2>
                    <p className="card-subtitle">Upload an image to get started</p>
                  </div>
                </div>
                <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                  <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.5 }}>🌿</div>
                  <p>Your plant analysis results will appear here</p>
                  <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                    Upload or capture a plant leaf image to detect diseases
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <FeaturesSection />
      </main>

      <Footer />
    </div>
  );
}

export default App;
