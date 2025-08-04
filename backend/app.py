from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64
import numpy as np
import os
import pathlib
import sys
import json
import uuid

# Import YOLOv5 from the original repository
try:
    # Clone YOLOv5 repo if not present
    if not os.path.exists('yolov5'):
        os.system('git clone https://github.com/ultralytics/yolov5.git')
    
    # Add yolov5 to path
    sys.path.insert(0, './yolov5')
    
    # Import YOLOv5 model
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
    
    YOLOV5_AVAILABLE = True
except ImportError as e:
    print(f"Error importing YOLOv5: {e}")
    print("YOLOv5 repository not found. Please clone it or use alternative solution.")
    YOLOV5_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Determine device for model loading
device = select_device('') if YOLOV5_AVAILABLE else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv5 model
model = None
if YOLOV5_AVAILABLE:
    try:
        if os.path.exists('best.pt'):
            model = DetectMultiBackend('best.pt', device=device)
            print("Loaded model: best.pt")
        elif os.path.exists('last.pt'):
            model = DetectMultiBackend('last.pt', device=device)
            print("Loaded model: last.pt")
        else:
            raise FileNotFoundError("Neither best.pt nor last.pt found in the current directory.")
        
        # Warm up the model
        if hasattr(model, 'warmup'):
            model.warmup(imgsz=(1, 3, 640, 640))
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Define class names mapping
CLASS_NAMES = {
    0: 'missing_hole',
    1: 'mouse_bite', 
    2: 'open_circuit',
    3: 'short',
    4: 'spur',
    5: 'spurious_copper'
}

def preprocess_image(image, target_size=640):
    """Preprocess image for YOLOv5 inference"""
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Letterbox resize (maintains aspect ratio)
    img = letterbox(img, target_size, stride=32, auto=True)[0]
    
    # Convert to torch tensor
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to 0-1
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img

@app.route('/')
def home():
    """Simple home route for health check."""
    return "PCB Defect Detector Backend is running!", 200

def expand_bounding_box(bbox, image_width, image_height, expansion_factor=0.05):
    """Expand bounding box by a given factor"""
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    expand_x = width * expansion_factor
    expand_y = height * expansion_factor
    
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(image_width, x2 + expand_x)
    new_y2 = min(image_height, y2 + expand_y)
    
    return new_x1, new_y1, new_x2, new_y2

def process_single_image(image_data, frontend_image_id, filename=None):
    """Process a single image for defect detection using YOLOv5"""
    try:
        # Handle different image input types
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            image = Image.open(image_data.stream).convert('RGB')
        
        original_width, original_height = image.size
        print(f"Processing image: {filename or frontend_image_id}, Original Size: {original_width}x{original_height}")
        
        # Preprocess image
        img_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
        
        detections = []
        
        # Process detections
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], (original_height, original_width)).round()
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                    confidence = float(conf)
                    class_id = int(cls)
                    class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
                    
                    # Optionally expand bounding box
                    x1_exp, y1_exp, x2_exp, y2_exp = expand_bounding_box(
                        (x1, y1, x2, y2), original_width, original_height
                    )
                    
                    detection_data = {
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": {
                            "x1": float(x1_exp),
                            "y1": float(y1_exp),
                            "x2": float(x2_exp),
                            "y2": float(y2_exp)
                        }
                    }
                    
                    detections.append(detection_data)
                    print(f"  Detection: {class_name} at ({x1_exp:.1f}, {y1_exp:.1f}) to ({x2_exp:.1f}, {y2_exp:.1f}) with confidence {confidence:.3f}")
        
        # Sort and limit detections
        max_detections_per_image = 20
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections_per_image]
        
        result = {
            "image_id": frontend_image_id,
            "predictions": detections,
            "image_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "total_detections": len(detections)
        }
        
        print(f"  Found {len(detections)} detections for {filename or frontend_image_id}")
        return result
        
    except Exception as e:
        print(f"Error processing image {filename or frontend_image_id}: {str(e)}")
        return {
            "image_id": frontend_image_id,
            "error": f"Processing failed: {str(e)}",
            "predictions": [],
            "image_dimensions": {"width": 0, "height": 0},
            "total_detections": 0
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    if model is None:
        return jsonify({"error": "Model not loaded. Check backend logs for details."}), 500

    results = []
    total_defects = 0
    processing_errors = []

    print("\n--- Starting prediction request ---")
    
    # Handle JSON array of image data
    if request.is_json and 'images_data' in request.json:
        images_data = request.json['images_data']
        print(f"Received {len(images_data)} images from JSON data.")
        for img_data in images_data:
            frontend_id = img_data.get('id', f"json_{uuid.uuid4().hex[:8]}")
            image_src = img_data.get('src')
            image_name = img_data.get('name', "JSON_Image")
            if image_src:
                result = process_single_image(image_src, frontend_id, image_name)
                results.append(result)
                if 'error' not in result:
                    total_defects += result['total_detections']
                else:
                    processing_errors.append(f"Image '{image_name}' (ID: {frontend_id}): {result['error']}")
            else:
                processing_errors.append(f"Image '{image_name}' (ID: {frontend_id}): Missing image source data.")

    # Handle FormData with multiple file uploads
    elif 'images' in request.files:
        files = request.files.getlist('images')
        image_metadata_json = request.form.get('image_metadata')
        image_metadata = []
        if image_metadata_json:
            try:
                image_metadata = json.loads(image_metadata_json)
            except json.JSONDecodeError:
                print("Warning: Could not decode image_metadata JSON from frontend.")

        print(f"Received {len(files)} uploaded files via FormData.")
        
        for i, file in enumerate(files):
            if file and file.filename:
                frontend_id = f"upload_{uuid.uuid4().hex[:8]}"
                frontend_name = file.filename
                
                if i < len(image_metadata):
                    meta = image_metadata[i]
                    frontend_id = meta.get('id', frontend_id)
                    frontend_name = meta.get('name', frontend_name)

                result = process_single_image(file, frontend_id, frontend_name)
                results.append(result)
                if 'error' not in result:
                    total_defects += result['total_detections']
                else:
                    processing_errors.append(f"Image '{frontend_name}' (ID: {frontend_id}): {result['error']}")
            else:
                processing_errors.append(f"Skipping empty or invalid file at index {i}.")
    
    # Handle single image upload (backward compatibility)
    elif 'image' in request.files:
        file = request.files['image']
        if file and file.filename:
            frontend_id = f"single_{uuid.uuid4().hex[:8]}"
            print(f"Received single image '{file.filename}' via old format.")
            result = process_single_image(file, frontend_id, file.filename)
            results.append(result)
            if 'error' not in result:
                total_defects += result['total_detections']
            else:
                processing_errors.append(f"Image '{file.filename}' (ID: {frontend_id}): {result['error']}")
    
    if not results:
        return jsonify({"error": "No valid images provided for detection."}), 400
    
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
            "model_names": CLASS_NAMES,
            "total_results": len(results)
        }
    }
    
    print(f"Processed {len(results)} images. Total defects found: {total_defects}")
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
