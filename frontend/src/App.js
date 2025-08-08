import { useState, useRef, useEffect } from 'react';
import { Camera, AlertCircle, CheckCircle, Loader2, X, Zap, Eye, Target, BarChart3, Download, Plus, Image as ImageIcon, ZoomIn, FileText, WifiOff } from 'lucide-react';

// The base URL for the backend API
const backendUrl = "http://13.51.242.26:8000";

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

  // Function to check the backend status
  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/health`);
      if (response.ok) {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    } catch (error) {
      console.error('Failed to fetch backend status:', error);
      setBackendStatus('offline');
    }
  };

  useEffect(() => {
    checkBackendStatus();
    // Check status every 60 seconds
    const interval = setInterval(checkBackendStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    const newImages = files.map((file) => ({
      file,
      url: URL.createObjectURL(file),
    }));
    setImages((prevImages) => [...prevImages, ...newImages]);
    setError(null);
  };

  const handleCameraCapture = () => {
    if (cameraStream) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        const file = new File([blob], `capture-${Date.now()}.png`, { type: 'image/png' });
        const newImage = {
          file,
          url: URL.createObjectURL(file),
        };
        setImages((prevImages) => [...prevImages, newImage]);
        setError(null);
      }, 'image/png');
      stopCamera();
    }
  };

  const startCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setCameraStream(stream);
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Error accessing camera. Please check your permissions.');
      setShowCamera(false);
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
      setShowCamera(false);
    }
  };

  const processImages = async () => {
    setLoading(true);
    setResults([]);
    setSummary({});
    setError(null);
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('images', image.file);
    });

    const startTime = performance.now();

    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const endTime = performance.now();
      setProcessingTime(((endTime - startTime) / 1000).toFixed(2));
      setResults(data);
      createSummary(data);
    } catch (e) {
      console.error('Error during image processing:', e);
      setError('Failed to process images. Please check the backend service and try again.');
    } finally {
      setLoading(false);
    }
  };

  const createSummary = (results) => {
    const defectCounts = {};
    results.forEach(result => {
      if (result.defects) {
        result.defects.forEach(defect => {
          defectCounts[defect.class] = (defectCounts[defect.class] || 0) + 1;
        });
      }
    });

    const totalDefects = Object.values(defectCounts).reduce((acc, count) => acc + count, 0);

    setSummary({
      totalImages: images.length,
      totalDefects,
      defectCounts,
    });
  };

  const handleClearImages = () => {
    images.forEach(image => URL.revokeObjectURL(image.url));
    setImages([]);
    setResults([]);
    setSummary({});
    setError(null);
    setSelectedImage(null);
  };

  const drawBoundingBoxes = (image, result) => {
    const canvas = document.createElement('canvas');
    const img = imageRefs.current[image.url];
    if (!img) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    if (result && result.defects && showBoundingBoxes) {
      result.defects.forEach(defect => {
        const [x1, y1, x2, y2] = defect.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const label = `${defect.class} (${(defect.score * 100).toFixed(1)}%)`;

        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);

        ctx.fillStyle = '#FF0000';
        ctx.font = '16px Arial';
        ctx.textBaseline = 'top';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x1, y1 - 20, textWidth + 4, 20);

        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 2, y1 - 18);
      });
    }

    const processedImageUrl = canvas.toDataURL();
    return processedImageUrl;
  };

  const handleDownloadResults = () => {
    if (results.length === 0) {
      return;
    }

    const data = JSON.stringify(results, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pcb_defect_results.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getDefectColor = (defectClass) => {
    const colors = {
      missing_hole: '#FF5733',
      mouse_bite: '#33FF57',
      open_circuit: '#3357FF',
      short: '#FF33A1',
      spur: '#A133FF',
      spurious_copper: '#FFC733',
    };
    return colors[defectClass] || '#666';
  };

  const handleImageClick = (image) => {
    setSelectedImage(image);
  };

  return (
    <div className="bg-gray-100 dark:bg-gray-900 min-h-screen font-sans antialiased text-gray-900 dark:text-gray-100 transition-colors duration-200">
      <header className="app-header bg-white dark:bg-gray-800 shadow-md p-4 sticky top-0 z-10 transition-colors duration-200">
        <div className="container mx-auto flex flex-col md:flex-row justify-between items-center">
          <div className="flex flex-col md:flex-row items-center gap-4">
            <h1 className="text-2xl md:text-3xl font-extrabold text-blue-600 dark:text-blue-400">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
                PCB Defect Detector
              </span>
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">Powered by YOLOv8</p>
          </div>
          <div className="flex items-center gap-2 mt-2 md:mt-0">
            <span
              className={`flex items-center gap-1 text-sm font-medium px-3 py-1 rounded-full ${
                backendStatus === 'online'
                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                  : backendStatus === 'offline'
                  ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                  : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
              }`}
            >
              {backendStatus === 'online' && <CheckCircle size={14} />}
              {backendStatus === 'offline' && <WifiOff size={14} />}
              {backendStatus === 'unknown' && <Loader2 size={14} className="animate-spin" />}
              <span>Backend: {backendStatus}</span>
            </span>
            <div className="tooltip-container relative">
              <AlertCircle size={16} className="text-gray-400 cursor-pointer" />
              <div className="tooltip-text absolute right-0 top-full mt-2 w-64 bg-gray-800 text-white text-xs rounded-lg p-2 text-center transform translate-x-1/2 -translate-y-full opacity-0 invisible transition-opacity duration-300">
                <span className="arrow-down absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full w-0 h-0 border-l-8 border-r-8 border-b-8 border-gray-800"></span>
                The backend service processes the images. If it's offline, please ensure the server is running.
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4 md:p-8 grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <aside className="sidebar lg:col-span-1 flex flex-col gap-6">
          <div className="controls-section bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-200">
            <h2 className="section-header text-xl font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center justify-between">
              Controls
            </h2>

            <div className="flex flex-col gap-4">
              <button
                onClick={() => fileInputRef.current.click()}
                className="control-button bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2"
              >
                <Plus size={20} />
                <span>Add Images</span>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageUpload}
                className="hidden"
                multiple
                accept="image/*"
              />

              <button
                onClick={showCamera ? stopCamera : startCamera}
                className="control-button bg-gray-700 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2"
              >
                <Camera size={20} />
                <span>{showCamera ? 'Stop Camera' : 'Use Camera'}</span>
              </button>

              <button
                onClick={processImages}
                className={`control-button bg-green-500 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2 ${
                  loading || images.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={loading || images.length === 0}
              >
                {loading ? <Loader2 size={20} className="animate-spin" /> : <Zap size={20} />}
                <span>Process Images</span>
              </button>

              <button
                onClick={handleClearImages}
                className={`control-button bg-red-500 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2 ${
                  images.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={images.length === 0}
              >
                <X size={20} />
                <span>Clear All</span>
              </button>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="toggle-boxes"
                  checked={showBoundingBoxes}
                  onChange={() => setShowBoundingBoxes(!showBoundingBoxes)}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                />
                <label htmlFor="toggle-boxes" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Show Bounding Boxes
                </label>
              </div>

              <button
                onClick={handleDownloadResults}
                className={`control-button bg-yellow-500 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2 ${
                  results.length === 0 ? 'opacity-50 cursor-not-allowed' : ''
                }`}
                disabled={results.length === 0}
              >
                <Download size={20} />
                <span>Download JSON</span>
              </button>
            </div>
          </div>

          <div className="legend-card bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-200">
            <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <FileText size={20} />
              Defect Legend
            </h2>
            <div className="legend-grid grid grid-cols-2 gap-3 text-sm">
              {Object.entries(defectDescriptions).map(([key, value]) => (
                <div key={key} className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: getDefectColor(key) }}
                  ></span>
                  <div className="flex-1">
                    <p className="font-semibold text-gray-800 dark:text-gray-200">{key.replace('_', ' ')}</p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">{value}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-lg shadow-md transition-colors duration-200" role="alert">
              <div className="flex items-center">
                <AlertCircle size={20} className="mr-2" />
                <p className="font-bold">Error</p>
              </div>
              <p className="mt-2 text-sm">{error}</p>
            </div>
          )}

          {showCamera && (
            <div className="camera-view bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-200 flex flex-col items-center gap-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200">Live Camera Feed</h2>
              <video ref={videoRef} autoPlay className="rounded-lg shadow-md max-w-full h-auto"></video>
              <canvas ref={canvasRef} className="hidden"></canvas>
              <button
                onClick={handleCameraCapture}
                className="control-button bg-blue-500 text-white font-semibold rounded-full p-3 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center justify-center gap-2"
              >
                <Camera size={20} />
                <span>Capture Image</span>
              </button>
            </div>
          )}

          <div className="image-gallery-section bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-200">
            <div className="section-header flex flex-col sm:flex-row items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                <ImageIcon size={20} />
                Image Gallery
              </h2>
              {loading && <p className="processing-time text-sm text-gray-500 dark:text-gray-400 animate-pulse">Processing images...</p>}
              {!loading && processingTime > 0 && (
                <p className="processing-time text-sm text-gray-500 dark:text-gray-400">
                  Processing Time: <span className="font-bold text-blue-500">{processingTime}s</span>
                </p>
              )}
            </div>

            {images.length === 0 && !loading && (
              <div className="text-center p-8 text-gray-500 dark:text-gray-400">
                <p className="text-lg">No images added yet.</p>
                <p className="mt-2 text-sm">Use the buttons on the left to add images for detection.</p>
              </div>
            )}

            <div className="images-grid grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 max-h-[calc(100vh-250px)] overflow-y-auto">
              {images.map((image, index) => (
                <div
                  key={index}
                  className="image-card relative bg-gray-50 dark:bg-gray-700 rounded-lg shadow-md overflow-hidden cursor-pointer group hover:shadow-xl transition-all duration-200"
                  onClick={() => handleImageClick(image)}
                >
                  <img
                    src={results[index] ? drawBoundingBoxes(image, results[index]) : image.url}
                    alt={`Uploaded PCB ${index + 1}`}
                    className="w-full h-auto object-cover transition-transform duration-200 group-hover:scale-105"
                    ref={el => imageRefs.current[image.url] = el}
                    onLoad={() => {
                        if (results[index] && imageRefs.current[image.url]) {
                            // Trigger a re-render to draw bounding boxes after image loads
                            setResults([...results]);
                        }
                    }}
                  />
                  {results[index] && results[index].defects && results[index].defects.length > 0 && (
                    <div className="absolute top-2 right-2 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg">
                      Defects: {results[index].defects.length}
                    </div>
                  )}
                  {results[index] && (!results[index].defects || results[index].defects.length === 0) && (
                    <div className="absolute top-2 right-2 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg">
                      No Defects
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {Object.keys(summary).length > 0 && (
            <div className="summary-card bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 transition-colors duration-200">
              <h2 className="section-header text-xl font-bold mb-4 text-gray-800 dark:text-gray-200 flex items-center gap-2">
                <BarChart3 size={20} />
                Detection Summary
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg shadow-inner">
                  <p className="text-sm font-semibold text-gray-600 dark:text-gray-400">Total Images</p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{summary.totalImages}</p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg shadow-inner">
                  <p className="text-sm font-semibold text-gray-600 dark:text-gray-400">Total Defects Found</p>
                  <p className="text-2xl font-bold text-red-600 dark:text-red-400">{summary.totalDefects}</p>
                </div>
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg shadow-inner">
                  <p className="text-sm font-semibold text-gray-600 dark:text-gray-400">Defects by Type</p>
                  <ul className="text-sm mt-2">
                    {Object.entries(summary.defectCounts).map(([defect, count]) => (
                      <li key={defect} className="flex items-center justify-between">
                        <span className="capitalize">{defect.replace('_', ' ')}:</span>
                        <span className="font-bold">{count}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedImage && (
            <div
              className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50"
              onClick={() => setSelectedImage(null)}
            >
              <div
                className="bg-white dark:bg-gray-800 p-4 rounded-2xl max-w-5xl max-h-[90vh] overflow-auto shadow-2xl relative"
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking on the image card
              >
                <button
                  onClick={() => setSelectedImage(null)}
                  className="absolute top-4 right-4 bg-gray-700 text-white rounded-full p-2 hover:bg-gray-600 transition-colors duration-200"
                >
                  <X size={24} />
                </button>
                <div className="relative">
                  <img
                    src={selectedImage.url}
                    alt="Selected PCB"
                    className="max-w-full max-h-[80vh] rounded-xl"
                  />
                  {results.find(res => res.filename === selectedImage.file.name)?.defects?.map((defect, index) => {
                    const [x1, y1, x2, y2] = defect.bbox;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    const label = `${defect.class} (${(defect.score * 100).toFixed(1)}%)`;

                    return (
                      <div
                        key={index}
                        className="absolute border-2 border-red-500"
                        style={{
                          left: `${x1}px`,
                          top: `${y1}px`,
                          width: `${width}px`,
                          height: `${height}px`,
                        }}
                      >
                        <div className="absolute top-0 left-0 transform -translate-y-full bg-red-500 text-white text-xs px-1 py-0.5 rounded-b-md">
                          {label}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <style>{`
        .tooltip-container:hover .tooltip-text {
          opacity: 1;
          visibility: visible;
          transform: translate(-50%, -100%);
        }
        .tooltip-container:hover .tooltip-text .arrow-down {
          border-b-color: #1a202c;
        }

        /* Mobile-first adjustments for better touch targets and readability */
        @media (max-width: 768px) {
          .container {
            padding: 1rem;
          }
          .control-button {
            width: 100%;
            display: flex;
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
