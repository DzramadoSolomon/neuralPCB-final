import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle, Loader2, X, Zap, Eye, Target, BarChart3, Download, RefreshCw } from 'lucide-react';

const PCBDefectDetector = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 640, height: 640 });
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [processingTime, setProcessingTime] = useState(0);
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  const defectColors = {
    missing_hole: '#FF6B6B',
    mouse_bite: '#4ECDC4',
    open_circuit: '#45B7D1',
    short: '#FFA07A',
    spur: '#98D8C8',
    spurious_copper: '#F7DC6F'
  };

  const defectDescriptions = {
    missing_hole: 'A hole that should be present in the PCB is missing',
    mouse_bite: 'Small semi-circular notches along the board edges',
    open_circuit: 'Break in the conductive path preventing current flow',
    short: 'Unintended connection between conductors',
    spur: 'Unwanted protrusion of copper material',
    spurious_copper: 'Excess copper material where it shouldn\'t be'
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
        setImagePreview(e.target.result);
        setDetections([]);
        setError(null);
        setProcessingTime(0);
      };
      reader.readAsDataURL(file);
    }
  };

  const detectDefects = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const formData = new FormData();
      const blob = await fetch(image).then(res => res.blob());
      formData.append('image', blob, 'pcb_image.png');

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      const endTime = performance.now();
      setProcessingTime(((endTime - startTime) / 1000).toFixed(2));

      if (response.ok) {
        setDetections(data.predictions || []);
        if (data.image_dimensions) {
          setImageDimensions(data.image_dimensions);
        }
      } else {
        setError(data.error || 'Detection failed');
      }
    } catch (err) {
      console.error(err);
      setError('Failed to connect to the backend. Make sure Flask server is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setImage(null);
    setImagePreview(null);
    setDetections([]);
    setError(null);
    setImageDimensions({ width: 640, height: 640 });
    setProcessingTime(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getBoundingBoxStyle = (bbox, imageElement) => {
    if (!imageElement) return {};
    
    const imageRect = imageElement.getBoundingClientRect();
    const scaleX = imageRect.width / imageDimensions.width;
    const scaleY = imageRect.height / imageDimensions.height;
    
    return {
      position: 'absolute',
      left: bbox.x1 * scaleX,
      top: bbox.y1 * scaleY,
      width: (bbox.x2 - bbox.x1) * scaleX,
      height: (bbox.y2 - bbox.y1) * scaleY,
      border: `3px solid ${defectColors[bbox.class] || '#FF0000'}`,
      backgroundColor: `${defectColors[bbox.class] || '#FF0000'}15`,
      borderRadius: '4px',
      pointerEvents: 'none',
      animation: 'pulse 2s infinite'
    };
  };

  const getDefectStats = () => {
    const stats = {};
    detections.forEach(detection => {
      stats[detection.class] = (stats[detection.class] || 0) + 1;
    });
    return stats;
  };

  const downloadReport = () => {
    const stats = getDefectStats();
    const report = {
      timestamp: new Date().toISOString(),
      processingTime: processingTime,
      totalDefects: detections.length,
      defectStats: stats,
      detections: detections.map(d => ({
        type: d.class,
        confidence: d.confidence,
        location: d.bbox
      }))
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pcb_defect_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      <div className="main-wrapper">
        {/* Header */}
        <div className="app-header">
          <div className="header-title-wrapper">
            <div className="header-icon">
              <Zap size={32} />
            </div>
            <h1 className="header-title">
              NEURAL PCB 
            </h1>
          </div>
          <p className="header-subtitle">
            Advanced AI-powered PCB quality control system. Upload your PCB image to detect manufacturing defects with high precision.
          </p>
        </div>

        {/* Main Content */}
        <div className="content-grid">
          {/* Upload and Image Section */}
          <div className="left-column">
            {/* Upload Section */}
            <div className="glass-card upload-section">
              <div className="section-header">
                <h2 className="section-title">
                  <Upload size={24} />
                  Image Upload
                </h2>
                {processingTime > 0 && (
                  <div className="processing-time">
                    Processing time: {processingTime}s
                  </div>
                )}
              </div>

              <div className="upload-controls">
                <label className="upload-button">
                  <Camera size={20} />
                  Choose PCB Image
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                  />
                </label>
                
                {image && (
                  <button
                    onClick={clearImage}
                    className="clear-button"
                  >
                    <X size={20} />
                    Clear
                  </button>
                )}

                {detections.length > 0 && (
                  <button
                    onClick={downloadReport}
                    className="report-button"
                  >
                    <Download size={20} />
                    Report
                  </button>
                )}
              </div>

              {error && (
                <div className="error-message">
                  <AlertCircle size={20} />
                  {error}
                </div>
              )}

              {/* Image Display */}
              {imagePreview && (
                <div className="image-display">
                  <div className="image-container">
                    <div className="image-wrapper">
                      <img
                        ref={imageRef}
                        src={imagePreview}
                        alt="PCB"
                        className="pcb-image"
                      />
                      
                      {/* Bounding boxes overlay */}
                      {showBoundingBoxes && detections.map((detection, index) => {
                        const imageElement = imageRef.current;
                        const boxStyle = getBoundingBoxStyle(detection.bbox, imageElement);
                        
                        return (
                          <div key={index} style={boxStyle} className="bounding-box">
                            <div 
                              className="bounding-box-label"
                              style={{ backgroundColor: defectColors[detection.class] || '#FF0000' }}
                            >
                              {detection.class ? detection.class.replace('_', ' ') : 'Unknown'} ({(detection.confidence * 100).toFixed(1)}%)
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Controls */}
                  <div className="image-controls">
                    <div className="checkbox-wrapper">
                      <input
                        type="checkbox"
                        checked={showBoundingBoxes}
                        onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                      />
                      <Eye size={16} />
                      <label>Show Bounding Boxes</label>
                    </div>

                    <button
                      onClick={detectDefects}
                      disabled={loading}
                      className="detect-button"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="animate-spin" size={20} />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Target size={20} />
                          Detect Defects
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="right-column">
            {/* Statistics */}
            <div className="stats-section">
              <h3 className="stats-title">
                <BarChart3 size={20} />
                Detection Summary
              </h3>
              
              <div className="stats-list">
                <div className="stat-item total">
                  <span className="stat-label">Total Defects</span>
                  <span className="stat-value">{detections.length}</span>
                </div>
                
                {Object.entries(getDefectStats()).map(([defect, count]) => (
                  <div key={defect} className={`stat-item defect-${defect}`}>
                    <div className="stat-label">
                      <div className="stat-color-dot" />
                      <span className="stat-text">{defect.replace('_', ' ')}</span>
                    </div>
                    <span className="stat-value">{count}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Detection Results */}
            <div className="results-section">
              <h3 className="results-title">
                <CheckCircle size={20} />
                Detection Results
              </h3>

              {detections.length === 0 ? (
                <div className="no-results">
                  <Target size={48} />
                  <div className="no-results-title">
                    {loading ? 'Analyzing PCB...' : 'No defects detected'}
                  </div>
                  <p className="no-results-subtitle">
                    {loading ? 'Please wait while we process your image' : 'Upload an image and click "Detect Defects" to analyze'}
                  </p>
                </div>
              ) : (
                <div className="results-list">
                  {detections.map((detection, index) => (
                    <div
                      key={index}
                      className={`detection-card defect-${detection.class}`}
                    >
                      <div className="detection-header">
                        <div className="detection-type">
                          <div className="detection-type-dot" />
                          <h4 className="detection-type-name">
                            {detection.class ? detection.class.replace('_', ' ') : 'Unknown'}
                          </h4>
                        </div>
                        <span className="detection-confidence">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <p className="detection-description">
                        {defectDescriptions[detection.class] || 'Unknown defect type'}
                      </p>
                      
                      <div className="detection-position">
                        Position: ({detection.bbox.x1.toFixed(0)}, {detection.bbox.y1.toFixed(0)}) - 
                        ({detection.bbox.x2.toFixed(0)}, {detection.bbox.y2.toFixed(0)})
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Legend */}
            <div className="legend-section">
              <h4 className="legend-title">Defect Types Legend</h4>
              <div className="legend-list">
                {Object.entries(defectColors).map(([defect, color]) => (
                  <div key={defect} className="legend-item">
                    <div 
                      className="legend-color"
                      style={{ backgroundColor: color }}
                    />
                    <span className="legend-text">
                      {defect.replace('_', ' ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PCBDefectDetector;