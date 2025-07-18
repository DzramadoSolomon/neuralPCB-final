from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
from PIL import Image
import io
import os
import pathlib
import sys
import numpy as np
from flask_cors import CORS

# Fix PosixPath issue (when loading Linux-trained models on Windows)
if isinstance(pathlib.Path(), pathlib.WindowsPath):
    pathlib.PosixPath = pathlib.WindowsPath

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Add yolov5 path so we can import DetectMultiBackend
yolov5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

try:
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print("Make sure YOLOv5 is properly installed and the path is correct")
    # Fallback to basic torch device selection
    def select_device(device=''):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names mapping (adjust according to your model's classes)
CLASS_NAMES = {
    0: 'missing_hole',
    1: 'mouse_bite', 
    2: 'open_circuit',
    3: 'short',
    4: 'spur',
    5: 'spurious_copper'
}

# Load YOLOv5 model using DetectMultiBackend
device = select_device('')  # Use select_device from YOLOv5 utils
model_path = 'last.pt'

try:
    model = DetectMultiBackend(model_path, device=device)
    model.eval()
    print("✅ YOLOv5 model loaded successfully.")
    print(f"Model device: {model.device}")
    print(f"Model names: {getattr(model, 'names', CLASS_NAMES)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Simple image preprocessing function
def preprocess_image(image, target_size=(640, 640)):
    """Simple preprocessing without letterboxing for testing"""
    # Resize image
    image_resized = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image_resized)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor /= 255.0  # Normalize to [0, 1]
    
    # Rearrange dimensions from HWC to CHW
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

@app.route('/')
def home():
    return "PCB Defect Detector Backend is running."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        # Load image
        image = Image.open(file.stream).convert('RGB')
        original_width, original_height = image.size
        
        # Preprocess image
        img_tensor = preprocess_image(image).to(device)
        
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Apply Non-Maximum Suppression
    conf_thres = 0.3
    iou_thres = 0.45
    
    try:
        # Try to use YOLOv5's NMS function
        pred = non_max_suppression(predictions, conf_thres, iou_thres)
    except:
        # Fallback to direct prediction processing if NMS fails
        pred = predictions
        if isinstance(pred, tuple):
            pred = pred[0]
        if not isinstance(pred, list):
            pred = [pred]
    
    detections = []
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Try to rescale boxes if scale_boxes is available
            try:
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], (original_height, original_width)).round()
            except:
                # Manual rescaling if scale_boxes fails
                scale_x = original_width / 640
                scale_y = original_height / 640
                det[:, 0] *= scale_x  # x1
                det[:, 1] *= scale_y  # y1
                det[:, 2] *= scale_x  # x2
                det[:, 3] *= scale_y  # y2
            
            # Process each detection
            for *xyxy, conf, cls in reversed(det):
                class_id = int(cls)
                confidence = float(conf)
                
                # Get class name
                class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
                
                # Convert coordinates to float
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                
                detections.append({
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

    # Sort by confidence and limit results
    max_detections = 10  # Increased from 5 to see more detections
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]

    # Debug information
    print(f"Total detections found: {len(detections)}")
    for det in detections:
        print(f"Class: {det['class']}, Confidence: {det['confidence']:.3f}")

    return jsonify({
        "predictions": detections,
        "image_dimensions": {
            "width": original_width,
            "height": original_height
        },
        "debug_info": {
            "total_detections": len(detections),
            "model_device": str(device),
            "model_names": getattr(model, 'names', CLASS_NAMES)
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)