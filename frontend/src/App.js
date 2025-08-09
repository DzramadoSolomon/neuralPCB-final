import { useState, useRef, useEffect } from 'react';
import { Camera, AlertCircle, CheckCircle, Loader2, X, Zap, Eye, Target, BarChart3, Download, Plus, Image as ImageIcon, ZoomIn, FileText, WifiOff } from 'lucide-react';

const defectDescriptions = {
  missing_hole: 'A hole that should be present in the PCB is missing',
  mouse_bite: 'Small semi-circular notches along the board edges',
  open_circuit: 'Break in the conductive path preventing current flow',
  short: 'Unintended connection between conductors',
  spur: 'Unwanted protrusion of copper material',
  spurious_copper: 'Excess copper material where it shouldn\'t be'
};

const PCBDefectDetector = () => {
  const [images, setImages] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [processingTime, setProcessingTime] = useState(0);
  const [summary, setSummary] = useState({});
  const [showCamera, setShowCamera] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [backendStatus, setBackendStatus] = useState('unknown'); // 'online', 'offline', 'unknown'
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRefs = useRef({});
  const modalImageRef = useRef(null);

  const defectColors = {
    missing_hole: '#FF6B6B',
    mouse_bite: '#4ECDC4',
    open_circuit: '#45B7D1',
    short: '#FFA07A',
    spur: '#98D8C8',
    spurious_copper: '#F7DC6F'
  };

  // Check backend status on component mount
  useEffect(() => {
    checkBackendStatus();
  }, []);

 const checkBackendStatus = async () => {
    try {
      const response = await fetch('https://appropriate-accuracy-suffering-d.trycloudflare.com/health', {
        method: 'GET',
        mode: 'cors',
      });
      
      if (response.ok) {
        const data = await response.json();
        setBackendStatus('online');
        console.log('Backend health check:', data);
      } else {
        setBackendStatus('offline');
      }
    } catch (err) {
      console.error('Backend health check failed:', err);
      setBackendStatus('offline');
    }
  };
  
  // Camera functions
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'environment'
        } 
      });
      setCameraStream(stream);
      setShowCamera(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setError('Could not access camera. Please check permissions.');
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setShowCamera(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      
      const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
      const newImage = {
        id: `camera_${Date.now()}`,
        src: imageDataUrl,
        name: `Camera_Capture_${new Date().toLocaleTimeString()}`,
        type: 'camera'
      };
      
      setImages(prev => [...prev, newImage]);
      stopCamera();
    }
  };

  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [cameraStream]);

  const handleMultipleImageUpload = (event) => {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const newImage = {
          id: `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          src: e.target.result,
          name: file.name,
          file: file,
          type: 'upload'
        };
        setImages(prev => [...prev, newImage]);
      };
      reader.readAsDataURL(file);
    });
    
    setResults([]);
    setError(null);
    setProcessingTime(0);
    setSummary({});
  };

  const removeImage = (imageId) => {
    setImages(prev => prev.filter(img => img.id !== imageId));
    setResults(prev => prev.filter(result => result.image_id !== imageId));
  };

  const detectDefects = async () => {
    if (images.length === 0) return;

    // Check backend status first
    if (backendStatus === 'offline') {
      setError('Backend server appears to be offline. Please wait a moment and try again, or contact support.');
      return;
    }

    setLoading(true);
    setError(null);
    setProcessingTime(0);

    const startTime = performance.now();
    
    try {
      let response;
      let requestBody;

      const backendUrl = 'https://appropriate-accuracy-suffering-d.trycloudflare.com/predict';

      const hasCameraImages = images.some(img => img.type === 'camera');

      if (hasCameraImages || images.some(img => img.type === 'upload' && !img.file)) {
        requestBody = {
          images_data: images.map(img => ({
            id: img.id,
            name: img.name,
            src: img.src
          }))
        };
        console.log('Sending JSON request to backend with', images.length, 'images');
        
        response = await fetch(backendUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          mode: 'cors',
          body: JSON.stringify(requestBody),
        });
      } else {
        const formData = new FormData();
        const imageMetadata = [];

        images.forEach((img) => {
          if (img.type === 'upload' && img.file) {
            formData.append('images', img.file);
            imageMetadata.push({ id: img.id, name: img.name });
          }
        });
        formData.append('image_metadata', JSON.stringify(imageMetadata));
        console.log('Sending FormData request to backend with', images.length, 'images');

        response = await fetch(backendUrl, {
          method: 'POST',
          mode: 'cors',
          body: formData,
        });
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log('Backend response:', data);

      const endTime = performance.now();
      setProcessingTime(((endTime - startTime) / 1000).toFixed(2));
      setResults(data.results || []);
      setSummary(data.summary || {});
      setBackendStatus('online'); // Backend responded successfully

    } catch (err) {
      console.error('Error during defect detection:', err);
      
      // Set backend status based on error type
      if (err.message.includes('Failed to fetch') || err.message.includes('ERR_FAILED')) {
        setBackendStatus('offline');
        setError('Unable to connect to the analysis server. The server may be starting up or experiencing issues. Please wait a moment and try again.');
      } else if (err.message.includes('502') || err.message.includes('503')) {
        setBackendStatus('offline');
        setError('The analysis server is temporarily unavailable (502/503 error). Please wait a few minutes and try again.');
      } else if (err.message.includes('CORS')) {
        setError('Connection blocked by browser security policy. Please contact support.');
      } else {
        setError(err.message || 'An unexpected error occurred during analysis.');
      }
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setImages([]);
    setResults([]);
    setError(null);
    setProcessingTime(0);
    setSummary({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    stopCamera();
  };

  // Bounding Box Overlay Component
  const BoundingBoxOverlay = ({ imageId, predictions, imageDimensions, isModal = false }) => {
    const [boxes, setBoxes] = useState([]);

    useEffect(() => {
      const imageElement = isModal ? modalImageRef.current : imageRefs.current[imageId];
      
      if (!showBoundingBoxes || !imageElement || !imageDimensions || !predictions || predictions.length === 0) {
        setBoxes([]);
        return;
      }

      const updateBoxes = () => {
        if (!imageElement.complete || imageElement.naturalWidth === 0 || imageDimensions.width === 0 || imageDimensions.height === 0) {
          return;
        }

        const naturalWidth = imageDimensions.width;
        const naturalHeight = imageDimensions.height;
        
        const imgRect = imageElement.getBoundingClientRect();
        const imgDisplayWidth = imgRect.width;
        const imgDisplayHeight = imgRect.height;

        let renderedImageWidth;
        let renderedImageHeight;
        let offsetX = 0;
        let offsetY = 0;

        const imageAspectRatio = naturalWidth / naturalHeight;
        const containerAspectRatio = imgDisplayWidth / imgDisplayHeight;

        if (containerAspectRatio > imageAspectRatio) {
          renderedImageHeight = imgDisplayHeight;
          renderedImageWidth = imgDisplayHeight * imageAspectRatio;
          offsetX = (imgDisplayWidth - renderedImageWidth) / 2;
        } else {
          renderedImageWidth = imgDisplayWidth;
          renderedImageHeight = imgDisplayWidth / imageAspectRatio;
          offsetY = (imgDisplayHeight - renderedImageHeight) / 2;
        }

        const scaleX = renderedImageWidth / naturalWidth;
        const scaleY = renderedImageHeight / naturalHeight;

        const newBoxes = predictions.map((detection, index) => {
          const box = detection.bbox || detection.location || {};
          
          const x1 = parseFloat(box.x1) || 0;
          const y1 = parseFloat(box.y1) || 0;
          const x2 = parseFloat(box.x2) || 0;
          const y2 = parseFloat(box.y2) || 0;

          const scaledBox = {
            id: index,
            left: (x1 * scaleX) + offsetX,
            top: (y1 * scaleY) + offsetY,
            width: (x2 - x1) * scaleX,
            height: (y2 - y1) * scaleY,
            class: detection.class || detection.type || 'unknown',
            confidence: detection.confidence || 0
          };

          return scaledBox;
        });

        setBoxes(newBoxes);
      };

      let timeoutId;
      const debouncedUpdate = () => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(updateBoxes, 100);
      };

      if (imageElement.complete && imageElement.naturalWidth > 0 && imageDimensions.width > 0) {
        updateBoxes();
      } else {
        imageElement.addEventListener('load', updateBoxes);
      }

      const resizeObserver = new ResizeObserver(debouncedUpdate);
      resizeObserver.observe(imageElement);
      
      window.addEventListener('resize', debouncedUpdate);

      return () => {
        clearTimeout(timeoutId);
        resizeObserver.disconnect();
        window.removeEventListener('resize', debouncedUpdate);
        imageElement.removeEventListener('load', updateBoxes);
      };
    }, [imageId, predictions, imageDimensions, isModal, showBoundingBoxes]);

    if (!showBoundingBoxes || boxes.length === 0) {
      return null;
    }

    return (
      <>
        {boxes.map((box) => (
          <div
            key={box.id}
            style={{
              position: 'absolute',
              left: `${box.left}px`,
              top: `${box.top}px`,
              width: `${box.width}px`,
              height: `${box.height}px`,
              border: `3px solid ${defectColors[box.class] || '#FF0000'}`,
              backgroundColor: `${defectColors[box.class] || '#FF0000'}20`,
              borderRadius: '3px',
              pointerEvents: 'none',
              zIndex: 10,
              boxSizing: 'border-box',
              animation: 'pulse 2s infinite'
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: '-25px',
                left: '0',
                backgroundColor: defectColors[box.class] || '#FF0000',
                color: '#fff',
                padding: '3px 8px',
                fontSize: '11px',
                fontWeight: 'bold',
                borderRadius: '3px',
                whiteSpace: 'nowrap',
                boxShadow: '0 2px 4px rgba(0,0,0,0.3)',
                maxWidth: '200px',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
            >
              {box.class.replace('_', ' ')} {(box.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </>
    );
  };

  // Image Modal Component
  const ImageModal = () => {
    if (!selectedImage) return null;

    const imageResult = results.find(r => r.image_id === selectedImage.id);

    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '20px'
      }} onClick={() => setSelectedImage(null)}>
        <div style={{
          position: 'relative',
          maxWidth: '90vw',
          maxHeight: '90vh',
          backgroundColor: 'white',
          borderRadius: '15px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column'
        }} onClick={(e) => e.stopPropagation()}>
          
          <div style={{
            padding: '15px 20px',
            backgroundColor: '#f8f9fa',
            borderBottom: '1px solid #e9ecef',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexShrink: 0
          }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600' }}>
              {selectedImage.name}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              {imageResult && (
                <span style={{
                  background: '#d1ecf1',
                  color: '#0c5460',
                  padding: '6px 12px',
                  borderRadius: '15px',
                  fontSize: '0.9rem',
                  fontWeight: '600'
                }}>
                  {imageResult.total_detections} defects found
                </span>
              )}
              <button
                onClick={() => setSelectedImage(null)}
                style={{
                  background: '#dc3545',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  transition: 'background-color 0.2s ease'
                }}
              >
                <X size={18} />
              </button>
            </div>
          </div>

          <div style={{
            position: 'relative',
            maxHeight: 'calc(90vh - 140px)',
            overflow: 'auto',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '20px',
            flexGrow: 1
          }}>
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <img
                ref={modalImageRef}
                src={selectedImage.src}
                alt={selectedImage.name}
                style={{
                  maxWidth: '100%',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                  borderRadius: '8px'
                }}
                onLoad={() => {
                  setTimeout(() => {
                    if (modalImageRef.current) {
                      const event = new Event('load');
                      modalImageRef.current.dispatchEvent(event);
                    }
                  }, 200);
                }}
              />
              
              {imageResult && imageResult.predictions && (
                <BoundingBoxOverlay
                  imageId={selectedImage.id}
                  predictions={imageResult.predictions}
                  imageDimensions={imageResult.image_dimensions}
                  isModal={true}
                />
              )}
            </div>
          </div>

          {imageResult && imageResult.predictions && imageResult.predictions.length > 0 && (
            <div style={{
              padding: '20px',
              backgroundColor: '#f8f9fa',
              borderTop: '1px solid #e9ecef',
              maxHeight: '200px',
              overflowY: 'auto',
              flexShrink: 0
            }}>
              <h4 style={{ margin: '0 0 15px 0', fontSize: '1rem', fontWeight: '600' }}>
                Detected Defects ({imageResult.predictions.length})
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '10px' }}>
                {imageResult.predictions.map((detection, index) => (
                  <div key={index} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px',
                    backgroundColor: 'white',
                    borderRadius: '8px',
                    border: `2px solid ${defectColors[detection.class] || '#FF0000'}`,
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                  }}>
                    <div style={{
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      backgroundColor: defectColors[detection.class] || '#FF0000',
                      flexShrink: 0
                    }} />
                    <div style={{ flex: 1 }}>
                      <div
                        style={{
                          fontWeight: '600',
                          fontSize: '0.9rem',
                          textTransform: 'capitalize',
                          marginBottom: '2px'
                        }}
                        title={defectDescriptions[detection.class] || 'Defect info not available'}
                      >
                        {detection.class?.replace('_', ' ')}
                      </div>
                      <div style={{ fontSize: '0.8rem', color: '#666' }}>
                        Confidence: {(detection.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const getOverallStats = () => {
    return summary.defect_breakdown || {};
  };

  const downloadReport = () => {
    const report = {
      timestamp: new Date().toISOString(),
      processingTime: processingTime,
      totalImages: images.length,
      totalDefects: summary.total_defects_found || 0,
      overallStats: getOverallStats(),
      imageResults: results.map(result => ({
        imageId: result.image_id,
        defectCount: result.total_detections,
        detections: result.predictions?.map(d => ({
          type: d.class,
          confidence: d.confidence,
          location: d.bbox || d.location
        })) || []
      }))
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pcb_batch_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadDocumentReport = () => {
    let documentContent = `PCB DEFECT DETECTION REPORT\n`;
    documentContent += `Generated on: ${new Date().toLocaleString()}\n`;
    documentContent += `Processing Time: ${processingTime}s\n`;
    documentContent += `Total Images Processed: ${images.length}\n`;
    documentContent += `Total Defects Found: ${summary.total_defects_found || 0}\n\n`;
    
    documentContent += `OVERALL DEFECT STATISTICS:\n`;
    documentContent += `${'-'.repeat(50)}\n`;
    const overallStats = getOverallStats();
    Object.entries(overallStats).forEach(([defect, count]) => {
      documentContent += `${defect.replace('_', ' ').toUpperCase()}: ${count}\n`;
    });
    documentContent += `\n`;

    documentContent += `DETAILED IMAGE ANALYSIS:\n`;
    documentContent += `${'='.repeat(50)}\n\n`;

    results.forEach((result, index) => {
      const image = images.find(img => img.id === result.image_id);
      const imageName = image ? image.name : `Image ${index + 1}`;
      
      documentContent += `Image ${index + 1}: ${imageName}\n`;
      documentContent += `${'-'.repeat(30)}\n`;
      documentContent += `Total Defects: ${result.total_detections}\n`;
      
      if (result.error) {
        documentContent += `Error: ${result.error}\n`;
      } else if (result.predictions && result.predictions.length > 0) {
        documentContent += `\nDetected Defects:\n`;
        
        result.predictions.forEach((detection, detIndex) => {
          const bbox = detection.bbox || detection.location || {};
          documentContent += `  ${detIndex + 1}. ${detection.class?.replace('_', ' ').toUpperCase()}\n`;
          documentContent += `     - Confidence Score: ${(detection.confidence * 100).toFixed(1)}%\n`;
          documentContent += `     - Position: (${Math.round(bbox.x1 || 0)}, ${Math.round(bbox.y1 || 0)}) to (${Math.round(bbox.x2 || 0)}, ${Math.round(bbox.y2 || 0)})\n`;
          documentContent += `     - Description: ${defectDescriptions[detection.class] || 'No description available'}\n`;
          documentContent += `\n`;
        });
      } else {
        documentContent += `No defects detected in this image.\n`;
      }
      
      documentContent += `\n`;
    });

    documentContent += `DEFECT TYPE DESCRIPTIONS:\n`;
    documentContent += `${'='.repeat(50)}\n`;
    Object.entries(defectDescriptions).forEach(([type, description]) => {
      documentContent += `${type.replace('_', ' ').toUpperCase()}:\n`;
      documentContent += `  ${description}\n\n`;
    });

    const blob = new Blob([documentContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pcb_defect_report_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };


  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
      color: '#333'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
        {/* Header Section */}
        <div style={{ textAlign: 'center', marginBottom: '30px', color: 'white' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '15px', marginBottom: '15px' }}>
            <div style={{ background: 'rgba(255,255,255,0.2)', borderRadius: '15px', padding: '15px', backdropFilter: 'blur(10px)' }}>
              <Zap size={32} />
            </div>
            <h1 style={{ 
              fontSize: '2.5rem', 
              fontWeight: '800', 
              margin: '0',
              background: 'linear-gradient(45deg, #fff, #f0f0f0)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}>
              NEURAL PCB
            </h1>
          </div>
          <p style={{ fontSize: '1.1rem', opacity: '0.9', maxWidth: '600px', margin: '0 auto', lineHeight: '1.6' }}>
            Advanced AI-powered PCB quality control system. Upload multiple PCB images or use camera to detect manufacturing defects.
          </p>
          
          {/* Backend Status Indicator */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '8px', 
            marginTop: '15px',
            padding: '8px 15px',
            background: backendStatus === 'online' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
            borderRadius: '20px',
            fontSize: '0.9rem',
            fontWeight: '600'
          }}>
            {backendStatus === 'online' ? (
              <>
                <CheckCircle size={16} />
                Analysis Server Online
              </>
            ) : backendStatus === 'offline' ? (
              <>
                <WifiOff size={16} />
                Analysis Server Offline
              </>
            ) : (
              <>
                <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                Checking Server Status...
              </>
            )}
            <button 
              onClick={checkBackendStatus}
              style={{
                background: 'rgba(255,255,255,0.2)',
                border: 'none',
                borderRadius: '4px',
                padding: '2px 6px',
                color: 'white',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '30px', alignItems: 'start' }} className="main-grid">
          {/* Left Column - Image Management & Display */}
          <div>
            {/* Controls Section */}
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.95)', 
              borderRadius: '20px', 
              padding: '25px', 
              backdropFilter: 'blur(20px)',
              marginBottom: '20px',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
            }} className="controls-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }} className="section-header">
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: '0', fontSize: '1.3rem', fontWeight: '700' }}>
                  <ImageIcon size={24} />
                  Image Management
                </h2>
                {processingTime > 0 && (
                  <div style={{ background: '#e6fffa', color: '#008080', padding: '8px 15px', borderRadius: '20px', fontSize: '0.9rem', fontWeight: '600' }} className="processing-time">
                    Processing: {processingTime}s
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '15px' }} className="button-group">
                <label style={{
                  display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
                  background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                  color: 'white', borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                  border: 'none', transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                }} className="action-button">
                  <Plus size={18} />
                  Add Images
                  <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleMultipleImageUpload} style={{ display: 'none' }} />
                </label>
                
                <button onClick={showCamera ? stopCamera : startCamera} style={{
                  display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
                  background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
                  color: '#2d3748', borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                  border: 'none', transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                }} className="action-button">
                  <Camera size={18} />
                  {showCamera ? 'Stop Camera' : 'Use Camera'}
                </button>
                
                {images.length > 0 && (
                  <>
                    <button onClick={clearAll} style={{
                      display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
                      background: 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)',
                      color: '#2d3748', borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                      border: 'none', transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                    }} className="action-button">
                      <X size={18} />
                      Clear All
                    </button>
                    
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem', color: '#4a5568' }} className="checkbox-label">
                      <input type="checkbox" checked={showBoundingBoxes} onChange={(e) => setShowBoundingBoxes(e.target.checked)} />
                      <Eye size={16} />
                      Show Bounding Boxes
                    </label>
                  </>
                )}

                {results.length > 0 && (
                  <>
                    <button onClick={downloadReport} style={{
                      display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
                      background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
                      color: '#2d3748', borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                      border: 'none', transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                    }} className="action-button">
                      <Download size={18} />
                      JSON Report
                    </button>

                    <button onClick={downloadDocumentReport} style={{
                      display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 18px',
                      background: 'linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)',
                      color: '#2d3748', borderRadius: '10px', cursor: 'pointer', fontWeight: '600',
                      border: 'none', transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                    }} className="action-button">
                      <FileText size={18} />
                      Text Report
                    </button>
                  </>
                )}
              </div>

              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: '#fed7d7', color: '#c53030', padding: '12px', borderRadius: '8px', marginBottom: '15px' }} className="error-message">
                  <AlertCircle size={20} />
                  <div>
                    <div style={{ fontWeight: '600', marginBottom: '4px' }}>Connection Error</div>
                    <div style={{ fontSize: '0.9rem' }}>{error}</div>
                    {backendStatus === 'offline' && (
                      <div style={{ fontSize: '0.8rem', marginTop: '4px', opacity: '0.8' }}>
                        Tip: Try refreshing the server status or wait a few minutes for the server to start up.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Camera Live View Section */}
              {showCamera && (
                <div style={{ marginBottom: '20px' }} className="camera-section">
                  <div style={{ background: '#000', borderRadius: '12px', overflow: 'hidden', marginBottom: '12px' }}>
                    <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: '300px', objectFit: 'cover' }} />
                  </div>
                  <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }} className="camera-controls">
                    <button onClick={capturePhoto} style={{
                      background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                      color: 'white', border: 'none', padding: '12px 20px', borderRadius: '10px',
                      fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px',
                      transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                    }} className="action-button">
                      <Camera size={18} />
                      Capture Photo
                    </button>
                  </div>
                </div>
              )}

              <canvas ref={canvasRef} style={{ display: 'none' }} />

              {/* Detect Button */}
              {images.length > 0 && (
                <div style={{ textAlign: 'center' }} className="detect-button-container">
                  <button 
                    onClick={detectDefects} 
                    disabled={loading || backendStatus === 'offline'} 
                    style={{
                      display: 'flex', alignItems: 'center', gap: '10px', padding: '15px 30px',
                      background: loading || backendStatus === 'offline' ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      color: 'white', borderRadius: '12px', 
                      cursor: loading || backendStatus === 'offline' ? 'not-allowed' : 'pointer',
                      fontWeight: '700', fontSize: '1.1rem', border: 'none', margin: '0 auto',
                      transition: 'background-color 0.3s ease'
                    }} className="detect-button">
                    {loading ? (
                      <>
                        <Loader2 style={{ animation: 'spin 1s linear infinite' }} size={20} />
                        Analyzing {images.length} image{images.length > 1 ? 's' : ''}...
                      </>
                    ) : backendStatus === 'offline' ? (
                      <>
                        <WifiOff size={20} />
                        Server Offline - Cannot Analyze
                      </>
                    ) : (
                      <>
                        <Target size={20} />
                        Detect Defects ({images.length} image{images.length > 1 ? 's' : ''})
                      </>
                    )}
                  </button>
                  {backendStatus === 'offline' && images.length > 0 && (
                    <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '8px', textAlign: 'center' }}>
                      The analysis server is currently unavailable. Please check the status above and try again.
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Images Grid Display */}
            {images.length > 0 && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                gap: '15px',
                maxHeight: '500px',
                overflowY: 'auto',
                padding: '15px',
                background: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '15px',
                boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
              }} className="images-grid">
                {images.map((image) => {
                  const imageResult = results.find(r => r.image_id === image.id);
                  return (
                    <div key={image.id} style={{
                      background: 'white',
                      borderRadius: '12px',
                      overflow: 'hidden',
                      boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                      cursor: 'pointer',
                      transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                    }} className="image-card">
                      <div style={{ padding: '12px', background: '#f8f9fa', borderBottom: '1px solid #e9ecef', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }} className="image-card-header">
                        <h4 style={{ margin: '0', fontSize: '0.85rem', fontWeight: '600', color: '#495057', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px' }}>
                          {image.name}
                        </h4>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }} className="image-card-actions">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedImage(image);
                            }}
                            style={{
                              background: '#e3f2fd',
                              color: '#1976d2',
                              border: 'none',
                              borderRadius: '6px',
                              padding: '4px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              transition: 'background-color 0.2s ease'
                            }}
                            title="View larger"
                          >
                            <ZoomIn size={14} />
                          </button>
                          <button onClick={(e) => {
                            e.stopPropagation();
                            removeImage(image.id);
                          }} style={{
                            background: '#f8d7da', color: '#721c24', border: 'none', borderRadius: '6px',
                            padding: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center',
                            transition: 'background-color 0.2s ease'
                          }}>
                            <X size={14} />
                          </button>
                        </div>
                      </div>
                      
                      <div 
                        style={{ position: 'relative', height: '150px', overflow: 'hidden' }}
                        onClick={() => setSelectedImage(image)}
                      >
                        <img
                          ref={el => imageRefs.current[image.id] = el}
                          src={image.src}
                          alt={image.name}
                          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          onLoad={() => {
                            setTimeout(() => {
                              const imageElement = imageRefs.current[image.id];
                              if (imageElement) {
                                const event = new Event('load');
                                imageElement.dispatchEvent(event);
                              }
                            }, 200);
                          }}
                        />
                        
                        {imageResult && imageResult.predictions && (
                          <BoundingBoxOverlay
                            imageId={image.id}
                            predictions={imageResult.predictions}
                            imageDimensions={imageResult.image_dimensions}
                            isModal={false}
                          />
                        )}
                        
                        {imageResult && (
                          <div style={{ 
                            position: 'absolute', 
                            top: '8px', 
                            right: '8px',
                            background: 'rgba(209, 236, 241, 0.9)', 
                            color: '#0c5460', 
                            padding: '4px 8px', 
                            borderRadius: '12px', 
                            fontSize: '0.75rem', 
                            fontWeight: '600',
                            backdropFilter: 'blur(4px)'
                          }}>
                            {imageResult.total_detections} defects
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Column - Summary & Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }} className="sidebar">
            {/* Summary Statistics Card */}
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.95)', 
              borderRadius: '15px', 
              padding: '20px', 
              backdropFilter: 'blur(20px)',
              boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
            }} className="summary-card">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '0 0 15px 0', fontSize: '1.1rem', fontWeight: '700' }}>
                <BarChart3 size={18} />
                Summary
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', borderRadius: '8px', fontWeight: '700' }}>
                  <span>Total Images</span>
                  <span>{summary.total_images_processed || 0}</span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', borderRadius: '8px', fontWeight: '700' }}>
                  <span>Total Defects</span>
                  <span>{summary.total_defects_found || 0}</span>
                </div>
                
                {/* Breakdown of defects by type */}
                {Object.entries(getOverallStats()).map(([defect, count]) => (
                  <div key={defect} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 10px', background: '#f8f9fa', borderRadius: '6px', borderLeft: `4px solid ${defectColors[defect]}` }}>
                    <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: defectColors[defect] }} />
                    <span style={{ fontSize: '0.85rem', textTransform: 'capitalize' }}>{defect.replace('_', ' ')}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Detection Results Card */}
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.95)', 
              borderRadius: '15px', 
              padding: '20px', 
              backdropFilter: 'blur(20px)',
              boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
            }} className="results-card">
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '0 0 15px 0', fontSize: '1.1rem', fontWeight: '700' }}>
                <CheckCircle size={18} />
                Results
              </h3>

              {results.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '30px 10px', color: '#6c757d' }} className="no-results-message">
                  <Target size={40} style={{ opacity: '0.5', marginBottom: '10px' }} />
                  <div style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '5px' }}>
                    {loading ? 'Analyzing...' : 'No results yet'}
                  </div>
                  <p style={{ fontSize: '0.85rem', margin: '0', lineHeight: '1.4' }}>
                    {loading ? 'Please wait while we process your images' : 'Upload images and click "Detect Defects"'}
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '400px', overflowY: 'auto' }} className="results-list">
                  {results.map((result, index) => (
                    <div key={result.image_id} style={{ border: '1px solid #e9ecef', borderRadius: '8px', padding: '12px' }} className="image-result-item">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <h4 style={{ margin: '0', fontSize: '0.9rem', fontWeight: '600' }}>
                          {images.find(img => img.id === result.image_id)?.name || `Image ${index + 1}`}
                        </h4>
                        <span style={{ background: '#d1ecf1', color: '#0c5460', padding: '4px 8px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: '600' }}>
                          {result.total_detections} defects
                        </span>
                      </div>
                      
                      {result.error ? (
                        <div style={{ color: '#dc3545', fontSize: '0.85rem' }} className="error-text">{result.error}</div>
                      ) : (
                        result.predictions && result.predictions.length > 0 ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {/* Show top 5 detections, then a "+ more" link */}
                            {result.predictions.slice(0, 5).map((detection, i) => (
                              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 8px', background: '#f8f9fa', borderRadius: '4px', borderLeft: `3px solid ${defectColors[detection.class]}` }}>
                                <span style={{ fontSize: '0.8rem', textTransform: 'capitalize' }}>{detection.class?.replace('_', ' ')}</span>
                                <span style={{ fontSize: '0.75rem', fontWeight: '600' }}>{(detection.confidence * 100).toFixed(0)}%</span>
                              </div>
                            ))}
                            {result.predictions.length > 5 && (
                              <div style={{ fontSize: '0.75rem', color: '#6c757d', textAlign: 'center' }}>
                                +{result.predictions.length - 5} more
                              </div>
                            )}
                          </div>
                        ) : (
                          <div style={{ fontSize: '0.85rem', color: '#6c757d' }}>No defects detected.</div>
                        )
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Defect Types Legend Card */}
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.95)', 
              borderRadius: '15px', 
              padding: '15px', 
              backdropFilter: 'blur(20px)',
              boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)'
            }} className="legend-card">
              <h4 style={{ margin: '0 0 12px 0', fontSize: '1rem', fontWeight: '700' }}>Defect Types</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }} className="legend-grid">
                {Object.entries(defectColors).map(([defect, color]) => (
                  <div key={defect} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px', background: '#f8f9fa', borderRadius: '4px' }} className="legend-item">
                    <div style={{ width: '12px', height: '12px', borderRadius: '3px', backgroundColor: color, flexShrink: 0 }} />
                    <span style={{ fontSize: '0.75rem', color: '#495057', textTransform: 'capitalize' }}>
                      {defect.replace('_', ' ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Image Modal (rendered outside the main grid for full-screen overlay) */}
        <ImageModal />
      </div>

      {/* Global Styles for Animations and Responsiveness */}
      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }
        
        /* Hover effects for buttons and labels */
        button:hover {
          transform: translateY(-1px) !important;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        
        label:hover {
          transform: translateY(-1px) !important;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        }
        
        /* Hover effect for image cards */
        .image-card:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        }
        
        /* Base styles for responsiveness */
        .main-grid {
          grid-template-columns: 1fr 350px; /* Default for larger screens */
        }

        .button-group {
          flex-direction: row; /* Default for larger screens */
          flex-wrap: wrap;
        }

        .legend-grid {
          grid-template-columns: 1fr 1fr; /* Default for larger screens */
        }

        /* Responsive adjustments for smaller screens */
        @media (max-width: 1200px) {
          .main-grid {
            grid-template-columns: 1fr !important; /* Single column layout for smaller desktops/tablets */
          }
        }
        
        @media (max-width: 768px) {
          .app-header h1 {
            font-size: 2rem !important;
          }
          .app-header p {
            font-size: 1rem !important;
            padding: 0 10px; /* Add some padding for text on small screens */
          }
          .button-group {
            flex-direction: column !important; /* Stack buttons vertically on mobile */
            align-items: stretch !important; /* Stretch buttons to full width */
            gap: 10px !important; /* Adjust gap for stacked items */
          }
          .action-button, .checkbox-label {
            width: 100%; /* Full width for buttons and labels */
            justify-content: center; /* Center content in buttons */
            padding: 12px 15px !important; /* Increase padding for better touch targets */
          }
          .legend-grid {
            grid-template-columns: 1fr !important; /* Single column for legend on mobile */
          }
          .images-grid {
            grid-template-columns: 1fr !important; /* Single column for images on mobile */
            max-height: none !important; /* Remove fixed height, allow content to dictate height */
            overflow-y: visible !important; /* Allow content to flow naturally */
          }
          .image-card {
            width: 100%; /* Ensure image cards take full width */
          }
          .sidebar {
            padding: 15px !important; /* Adjust sidebar padding for mobile */
          }
          .controls-section, .summary-card, .results-card, .legend-card {
            padding: 15px !important; /* Adjust card padding for mobile */
          }
          .section-header {
            flex-direction: column; /* Stack header elements */
            align-items: flex-start;
            gap: 10px;
          }
          .processing-time {
            width: 100%;
            text-align: center;
          }
        }

        @media (max-width: 480px) {
          .app-header h1 {
            font-size: 1.8rem !important;
          }
          .app-header p {
            font-size: 0.9rem !important;
          }
        }
      `}</style>
    </div>
  );
};

export default PCBDefectDetector;
