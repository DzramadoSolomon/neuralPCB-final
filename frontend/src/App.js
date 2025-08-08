import { useState, useRef, useEffect } from 'react';
import { Camera, AlertCircle, CheckCircle, Loader2, X, Zap, Eye, Target, BarChart3, Download, Plus, Image as ImageIcon, ZoomIn, FileText } from 'lucide-react';

// IMPORTANT: Update this API_URL to match your backend server's address
const API_URL = 'http://13.51.242.26:8000';
const HEALTH_URL = `${API_URL}/health`;
const PREDICT_URL = `${API_URL}/predict`;

const defectDescriptions = {
  missing_hole: 'A hole that should be present in the PCB is missing',
  mouse_bite: 'Small semi-circular notches along the board edges',
  open_circuit: 'Break in the conductive path preventing current flow',
  short: 'Unintended connection between conductors',
  spur: 'Unwanted protrusion of copper material',
  spurious_copper: 'Excess copper material where it shouldn\\'t be'
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
    // Set up an interval to check the backend status every 10 seconds
    const interval = setInterval(checkBackendStatus, 10000);
    // Cleanup the interval when the component unmounts
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch(HEALTH_URL, {
        method: 'GET',
        mode: 'cors',
      });
      
      if (response.ok) {
        setBackendStatus('online');
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
        
        response = await fetch(PREDICT_URL, {
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

        response = await fetch(PREDICT_URL, {
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
              borderRadius: '4px',
              zIndex: 10,
              pointerEvents: 'none'
            }}
          >
            <div
              className="absolute text-xs text-white p-1 rounded-sm -top-6 left-0"
              style={{
                backgroundColor: `${defectColors[box.class] || '#FF0000'}`,
                whiteSpace: 'nowrap',
                fontWeight: 'bold'
              }}
            >
              {`${box.class} (${(box.confidence * 100).toFixed(1)}%)`}
            </div>
          </div>
        ))}
      </>
    );
  };

  // Helper function to get image dimensions
  const getImageDimensions = (imageId) => {
    const imageResult = results.find(r => r.image_id === imageId);
    if (imageResult && imageResult.image_dimensions) {
      return imageResult.image_dimensions;
    }
    const uploadedImage = images.find(img => img.id === imageId);
    if (uploadedImage) {
      const img = new window.Image();
      img.src = uploadedImage.src;
      return { width: img.naturalWidth, height: img.naturalHeight };
    }
    return null;
  };
  
  const handleDownloadResults = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Image Name,Defect Type,Confidence,Bounding Box (x1,y1,x2,y2)\n"
      + results.flatMap(result =>
          (result.predictions || []).map(p =>
            `${result.image_name},${p.class},${p.confidence.toFixed(4)},"${p.bbox.x1},${p.bbox.y1},${p.bbox.x2},${p.bbox.y2}"`
          )
        ).join('\n');
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'pcb_defect_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadSummary = () => {
    const summaryText = `
PCB Defect Detection Summary Report
===================================
Total Images Processed: ${summary.total_images_processed}
Total Defects Found: ${summary.total_defects_found}
Processing Time: ${processingTime} seconds

Defect Breakdown:
${Object.entries(summary.defect_breakdown || {}).map(([defect, count]) => `- ${defect}: ${count}`).join('\n')}

Processing Errors:
${(summary.processing_errors || []).length > 0 ? summary.processing_errors.map(err => `- ${err}`).join('\n') : 'None'}

Detailed Results:
${results.map(r => `
---
Image: ${r.image_name}
Image ID: ${r.image_id}
Defects Found: ${r.total_detections}
Predictions:
${(r.predictions || []).map(p => 
  `  - Class: ${p.class}, Confidence: ${(p.confidence * 100).toFixed(1)}%, Bbox: [${p.bbox.x1}, ${p.bbox.y1}, ${p.bbox.x2}, ${p.bbox.y2}]`
).join('\n')}
`).join('\n')}
    `;
    
    const blob = new Blob([summaryText], { type: 'text/plain;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'pcb_defect_summary.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const Modal = ({ image, onClose }) => {
    const defectResult = results.find(r => r.image_id === image.id);
    const predictions = defectResult ? defectResult.predictions : [];
    const imageDimensions = getImageDimensions(image.id);
    
    return (
      <div 
        onClick={onClose} 
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900 bg-opacity-75 backdrop-blur-sm"
      >
        <div 
          onClick={e => e.stopPropagation()} 
          className="relative max-w-full max-h-[90vh] bg-white rounded-lg shadow-xl overflow-hidden flex flex-col"
        >
          <div className="flex justify-between items-center p-4 bg-gray-100 border-b">
            <h3 className="text-xl font-bold text-gray-800">{image.name}</h3>
            <button onClick={onClose} className="p-1 rounded-full hover:bg-gray-200 transition-colors">
              <X className="w-6 h-6 text-gray-600" />
            </button>
          </div>
          <div className="relative p-4 flex-1 flex items-center justify-center overflow-auto">
            <img 
              ref={modalImageRef}
              src={image.src} 
              alt={image.name} 
              className="max-w-full max-h-full rounded-lg object-contain"
            />
            {imageDimensions && (
              <BoundingBoxOverlay 
                imageId={image.id}
                predictions={predictions} 
                imageDimensions={imageDimensions}
                isModal={true}
              />
            )}
          </div>
        </div>
      </div>
    );
  };

  const getDefectCount = (imageId) => {
    const result = results.find(r => r.image_id === imageId);
    return result ? result.total_detections : 0;
  };
  
  const getBackgroundColor = (imageId) => {
    const result = results.find(r => r.image_id === imageId);
    if (!result) return 'bg-gray-100';
    if (result.total_detections === 0) return 'bg-emerald-50';
    if (result.total_detections > 0) return 'bg-red-50';
    return 'bg-gray-100';
  }

  const getIcon = (imageId) => {
    const result = results.find(r => r.image_id === imageId);
    if (!result) return null;
    if (result.total_detections === 0) return <CheckCircle className="text-emerald-500"/>;
    if (result.total_detections > 0) return <AlertCircle className="text-red-500"/>;
    return null;
  }
  
  return (
    <div className="font-sans antialiased text-gray-800 bg-gray-50 min-h-screen">
      {/* Main Content Grid */}
      <div className="min-h-screen grid lg:grid-cols-[1fr_350px] gap-0">
        
        {/* Main Content Area */}
        <div className="flex flex-col p-6 lg:p-10 bg-white shadow-xl rounded-b-xl lg:rounded-r-none">
          
          {/* Header */}
          <header className="app-header mb-8 pb-4 border-b-2 border-gray-100">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h1 className="text-3xl lg:text-4xl font-extrabold text-gray-900 leading-tight mb-2">PCB Defect Detector</h1>
                <p className="text-md text-gray-500">
                  Analyze images of Printed Circuit Boards to detect common defects.
                </p>
              </div>
              <div className="ml-4">
                <div 
                  className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold transition-colors duration-300 ${
                    backendStatus === 'online' ? 'bg-emerald-100 text-emerald-800' : 
                    backendStatus === 'offline' ? 'bg-rose-100 text-rose-800' : 'bg-yellow-100 text-yellow-800'
                  }`}
                >
                  {backendStatus === 'online' && <CheckCircle className="w-4 h-4" />}
                  {backendStatus === 'offline' && <X className="w-4 h-4" />}
                  {backendStatus === 'unknown' && <Loader2 className="w-4 h-4 animate-spin" />}
                  <span>Server: {backendStatus.charAt(0).toUpperCase() + backendStatus.slice(1)}</span>
                </div>
              </div>
            </div>
          </header>

          {/* Controls */}
          <div className="flex-grow flex flex-col">
            <div className="controls-section mb-6 p-5 bg-gray-100 rounded-xl shadow-inner">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-gray-700">
                <Plus className="w-5 h-5"/> Add Images
              </h2>
              <div className="flex flex-col sm:flex-row gap-3">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleMultipleImageUpload}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  id="file-input"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="flex-1 flex justify-center items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:bg-blue-300"
                >
                  <ImageIcon className="w-5 h-5"/> Select Images
                </button>
                <button
                  onClick={startCamera}
                  disabled={loading}
                  className="flex-1 flex justify-center items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-emerald-600 text-white font-semibold hover:bg-emerald-700 disabled:bg-emerald-300"
                >
                  <Camera className="w-5 h-5"/> Use Camera
                </button>
              </div>
            </div>
            
            {/* Camera View */}
            {showCamera && (
              <div className="relative mb-6 p-5 bg-gray-100 rounded-xl shadow-inner flex flex-col items-center">
                <h3 className="text-lg font-bold mb-2">Live Camera Feed</h3>
                <video ref={videoRef} autoPlay playsInline className="w-full max-h-[60vh] rounded-lg shadow-md border-2 border-gray-300"></video>
                <div className="mt-4 flex gap-4">
                  <button onClick={capturePhoto} className="flex items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-indigo-600 text-white font-semibold hover:bg-indigo-700">
                    <Camera className="w-5 h-5"/> Capture Photo
                  </button>
                  <button onClick={stopCamera} className="flex items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-gray-400 text-white font-semibold hover:bg-gray-500">
                    <X className="w-5 h-5"/> Cancel
                  </button>
                </div>
              </div>
            )}
            
            {/* Image Grid */}
            {images.length > 0 && (
              <div className="flex-1 overflow-hidden flex flex-col">
                <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-gray-700">
                  <Eye className="w-5 h-5"/> Images for Analysis
                </h2>
                <div className="images-grid grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pb-5 overflow-y-auto custom-scrollbar-thin flex-1">
                  {images.map((image) => (
                    <div 
                      key={image.id}
                      className={`image-card relative w-full h-48 rounded-xl shadow-lg overflow-hidden transition-all duration-300 transform hover:scale-[1.02] cursor-pointer ${getBackgroundColor(image.id)}`}
                      onClick={() => setSelectedImage(image)}
                    >
                      <img 
                        ref={el => imageRefs.current[image.id] = el}
                        src={image.src} 
                        alt={image.name} 
                        className="w-full h-full object-contain p-4"
                        onLoad={() => {
                          const imgDim = getImageDimensions(image.id);
                          if (imgDim) {
                            const event = new Event('resize');
                            window.dispatchEvent(event);
                          }
                        }}
                      />
                      
                      <div className="absolute top-0 right-0 p-2 z-20">
                        <button onClick={(e) => { e.stopPropagation(); removeImage(image.id); }} className="text-white bg-red-500 rounded-full p-1.5 shadow-md hover:bg-red-600 transition-colors">
                          <X className="w-4 h-4"/>
                        </button>
                      </div>

                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-gray-900 to-transparent flex justify-between items-center text-white z-20">
                        <span className="text-sm font-semibold truncate">{image.name}</span>
                        <div className="flex items-center gap-2">
                          {getIcon(image.id)}
                          <span className="text-lg font-bold">
                            {getDefectCount(image.id)}
                          </span>
                        </div>
                      </div>
                      
                      {getDefectCount(image.id) > 0 && showBoundingBoxes && (
                        <BoundingBoxOverlay 
                          imageId={image.id}
                          predictions={results.find(r => r.image_id === image.id)?.predictions || []}
                          imageDimensions={getImageDimensions(image.id)}
                        />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Action Buttons */}
            {images.length > 0 && (
              <div className="mt-6 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={detectDefects}
                  disabled={loading || backendStatus === 'offline'}
                  className="flex-1 flex justify-center items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-indigo-600 text-white font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:text-gray-500"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin"/> Processing...
                    </>
                  ) : (
                    <>
                      <Target className="w-5 h-5"/> Start Detection
                    </>
                  )}
                </button>
                <button
                  onClick={clearAll}
                  className="flex-1 flex justify-center items-center gap-2 px-6 py-3 rounded-lg shadow-md transition-all duration-300 bg-gray-400 text-white font-semibold hover:bg-gray-500"
                >
                  <X className="w-5 h-5"/> Clear All
                </button>
              </div>
            )}
            
            {/* Error Message */}
            {error && (
              <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg shadow-sm" role="alert">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5"/>
                  <span className="font-medium">{error}</span>
                </div>
              </div>
            )}

            {/* Canvas for capturing photos */}
            <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
            
          </div>
        </div>

        {/* Sidebar */}
        <div className="sidebar flex flex-col p-6 lg:p-10 bg-gray-100 rounded-t-xl lg:rounded-l-none">
          <div className="flex-1 flex flex-col gap-6">
            
            {/* Summary Card */}
            <div className="summary-card p-5 bg-white rounded-xl shadow-md">
              <div className="section-header flex items-center justify-between pb-4 border-b border-gray-200 mb-4">
                <h2 className="text-xl font-bold text-gray-700 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5"/> Summary
                </h2>
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 cursor-pointer transition-all duration-300 px-3 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-600 shadow-sm hover:bg-gray-200">
                    <input
                      type="checkbox"
                      checked={showBoundingBoxes}
                      onChange={() => setShowBoundingBoxes(!showBoundingBoxes)}
                      className="form-checkbox text-blue-600 rounded-md"
                    />
                    <span>Show Boxes</span>
                  </label>
                  <button onClick={handleDownloadSummary} disabled={!summary.total_images_processed} className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold text-white bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 disabled:text-gray-500 transition-colors">
                    <FileText className="w-3.5 h-3.5"/> Summary
                  </button>
                  <button onClick={handleDownloadResults} disabled={!summary.total_images_processed} className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold text-white bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 disabled:text-gray-500 transition-colors">
                    <Download className="w-3.5 h-3.5"/> CSV
                  </button>
                </div>
              </div>
              <div className="text-sm grid grid-cols-2 gap-y-4">
                <div className="font-semibold text-gray-500">Images Processed:</div>
                <div className="text-right font-bold text-gray-800">
                  {summary.total_images_processed || 0}
                </div>
                
                <div className="font-semibold text-gray-500">Total Defects:</div>
                <div className={`text-right font-bold ${summary.total_defects_found > 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                  {summary.total_defects_found || 0}
                </div>
                
                {processingTime > 0 && (
                  <>
                    <div className="font-semibold text-gray-500">Processing Time:</div>
                    <div className="text-right font-bold text-gray-800">
                      {processingTime} s
                    </div>
                  </>
                )}
              </div>
            </div>
            
            {/* Defect Breakdown Card */}
            {summary.total_defects_found > 0 && (
              <div className="results-card p-5 bg-white rounded-xl shadow-md">
                <h3 className="text-lg font-bold pb-4 border-b border-gray-200 text-gray-700">Defect Breakdown</h3>
                <ul className="mt-4 space-y-3">
                  {Object.entries(summary.defect_breakdown || {})
                    .sort(([, countA], [, countB]) => countB - countA)
                    .map(([defect, count]) => (
                    <li key={defect} className="flex justify-between items-center text-sm">
                      <span className="font-medium text-gray-600 capitalize">
                        {defect.replace('_', ' ')}
                      </span>
                      <span className={`px-2 py-0.5 rounded-full font-bold ${defect === 'short' || defect === 'open_circuit' ? 'bg-red-100 text-red-600' : 'bg-yellow-100 text-yellow-600'}`}>
                        {count}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {/* Legend Card */}
            <div className="legend-card p-5 bg-white rounded-xl shadow-md flex-1">
              <h3 className="text-lg font-bold pb-4 border-b border-gray-200 text-gray-700">Defect Legend</h3>
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                {Object.entries(defectDescriptions).map(([defect, description]) => (
                  <div key={defect} className="flex items-start gap-3">
                    <div
                      className="w-4 h-4 rounded-full flex-shrink-0 mt-1"
                      style={{ backgroundColor: defectColors[defect] }}
                    ></div>
                    <div className="flex-1">
                      <p className="font-bold text-sm leading-none capitalize">
                        {defect.replace('_', ' ')}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* About Section */}
            <footer className="mt-auto pt-6 text-sm text-gray-400 text-center border-t border-gray-200">
              <p>Powered by YOLOv5 and React.js</p>
            </footer>

          </div>
        </div>
      </div>
      
      {/* Modal for Full-screen Image View */}
      {selectedImage && (
        <Modal 
          image={selectedImage} 
          onClose={() => setSelectedImage(null)} 
        />
      )}

      {/* Tailwind and custom styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        body {
          font-family: 'Inter', sans-serif;
        }

        .custom-scrollbar-thin::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar-thin::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar-thin::-webkit-scrollbar-thumb {
          background: #888;
          border-radius: 10px;
        }
        .custom-scrollbar-thin::-webkit-scrollbar-thumb:hover {
          background: #555;
        }

        /* Loading animation for processing button */
        .animate-pulse-light {
          animation: pulse-light 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        @keyframes pulse-light {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 1; }
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
        
        /* Responsive adjustments for smaller screens */
        @media (max-width: 1200px) {
          /* Change main grid to single column layout */
          div[style*="gridTemplateColumns: 1fr 350px"] {
            grid-template-columns: 1fr !important;
          }
        }
        
        @media (max-width: 768px) {
          .app-header h1 {
            font-size: 2rem !important;
          }
          .app-header p {
            font-size: 1rem !important;
          }
          div[style*="display: flex; gap: 12px; flex-wrap: wrap"] {
            flex-direction: column !important; /* Stack buttons vertically */
          }
          .controls-section div {
            flex-direction: column;
            gap: 12px;
          }
          .controls-section button {
            width: 100%;
            display: flex; /* Ensure button content is centered */
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
