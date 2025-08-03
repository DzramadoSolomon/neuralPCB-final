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
import uuid
import base64

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

def expand_bounding_box(bbox, image_width, image_height, expansion_factor=0.15):
    """Expand bounding box by a factor while keeping it within image bounds"""
    x1, y1, x2, y2 = bbox
    
    # Calculate current width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion amounts
    expand_x = width * expansion_factor
    expand_y = height * expansion_factor
    
    # Expand the box
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(image_width, x2 + expand_x)
    new_y2 = min(image_height, y2 + expand_y)
    
    return new_x1, new_y1, new_x2, new_y2

def process_single_image(image_data, image_id):
    """Process a single image and return detections"""
    try:
        # Handle different image input types
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Base64 encoded image from camera
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # File upload
            image = Image.open(image_data.stream).convert('RGB')
        
        original_width, original_height = image.size
        
        # Preprocess image
        img_tensor = preprocess_image(image).to(device)
        
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
                    
                    # Convert coordinates to float and expand bounding box
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    x1_exp, y1_exp, x2_exp, y2_exp = expand_bounding_box(
                        (x1, y1, x2, y2), original_width, original_height
                    )
                    
                    detections.append({
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": x1_exp,
                            "y1": y1_exp,
                            "x2": x2_exp,
                            "y2": y2_exp
                        },
                        "location": {
                            "x1": x1_exp,
                            "y1": y1_exp,
                            "x2": x2_exp,
                            "y2": y2_exp
                        }
                    })

        # Sort by confidence and limit results
        max_detections = 10
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections]

        return {
            "image_id": image_id,
            "predictions": detections,
            "image_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "total_detections": len(detections)
        }
        
    except Exception as e:
        return {
            "image_id": image_id,
            "error": f"Processing failed: {str(e)}",
            "predictions": [],
            "image_dimensions": {"width": 0, "height": 0},
            "total_detections": 0
        }

@app.route('/')
def home():
    return "PCB Defect Detector Backend is running."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Handle both single and multiple image uploads
    results = []
    total_defects = 0
    processing_errors = []
    
    # Check for camera data (base64 encoded)
    if 'camera_data' in request.json if request.is_json else False:
        try:
            camera_data = request.json['camera_data']
            image_id = f"camera_{uuid.uuid4().hex[:8]}"
            result = process_single_image(camera_data, image_id)
            results.append(result)
            if 'error' not in result:
                total_defects += result['total_detections']
            else:
                processing_errors.append(f"Camera image: {result['error']}")
        except Exception as e:
            processing_errors.append(f"Camera processing error: {str(e)}")
    
    # Check for file uploads
    if 'images' in request.files:
        files = request.files.getlist('images')
        for i, file in enumerate(files):
            if file and file.filename:
                image_id = f"upload_{i}_{uuid.uuid4().hex[:8]}"
                result = process_single_image(file, image_id)
                results.append(result)
                if 'error' not in result:
                    total_defects += result['total_detections']
                else:
                    processing_errors.append(f"Image {i+1}: {result['error']}")
    
    # Handle single image upload (backward compatibility)
    elif 'image' in request.files:
        file = request.files['image']
        if file:
            image_id = f"single_{uuid.uuid4().hex[:8]}"
            result = process_single_image(file, image_id)
            results.append(result)
            if 'error' not in result:
                total_defects += result['total_detections']
            else:
                processing_errors.append(result['error'])
    
    if not results:
        return jsonify({"error": "No images provided"}), 400
    
    # Generate summary statistics
    defect_summary = {}
    for result in results:
        if 'error' not in result:
            for detection in result['predictions']:
                defect_type = detection['class']
                defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
    
    response_data = {
        "results": results,
        "summary": {
            "total_images_processed": len(results),
            "total_defects_found": total_defects,
            "defect_breakdown": defect_summary,
            "processing_errors": processing_errors
        },
        "debug_info": {
            "model_device": str(device),
            "model_names": getattr(model, 'names', CLASS_NAMES)
        }
    }
    
    # Debug information
    print(f"Processed {len(results)} images")
    print(f"Total defects found: {total_defects}")
    for defect_type, count in defect_summary.items():
        print(f"{defect_type}: {count}")
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)