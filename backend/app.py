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
import json # Import json for handling metadata

# Fix PosixPath issue (when loading Linux-trained models on Windows)
if isinstance(pathlib.Path(), pathlib.WindowsPath):
    pathlib.PosixPath = pathlib.WindowsPath

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Add yolov5 path so we can import DetectMultiBackend
# Assumes yolov5 repository is cloned as a sibling directory or within the same directory
yolov5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

try:
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print("Make sure YOLOv5 is properly installed and the path is correct.")
    print("You might need to run: pip install -r yolov5/requirements.txt")
    # Fallback to basic torch device selection if YOLOv5 imports fail
    def select_device(device=''):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names mapping (adjust according to your model's classes)
# These should match the classes your YOLOv5 model was trained on
CLASS_NAMES = {
    0: 'missing_hole',
    1: 'mouse_bite', 
    2: 'open_circuit',
    3: 'short',
    4: 'spur',
    5: 'spurious_copper'
}

# Load YOLOv5 model using DetectMultiBackend
device = select_device('')  # Automatically selects CUDA if available, else CPU
# Ensure 'last.pt' (your trained YOLOv5 model) is in the same directory as app.py
model_path = 'best.pt'

try:
    model = DetectMultiBackend(model_path, device=device)
    model.eval() # Set model to evaluation mode
    print("✅ YOLOv5 model loaded successfully.")
    print(f"Model device: {model.device}")
    print(f"Model names: {getattr(model, 'names', CLASS_NAMES)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None # Set model to None if loading fails to prevent further errors

# Simple image preprocessing function
def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocesses a PIL Image for YOLOv5 inference.
    Resizes the image and normalizes pixel values.
    """
    # Resize image to target_size (e.g., 640x640)
    image_resized = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image_resized)
    
    # Convert to tensor and normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    # Rearrange dimensions from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    
    # Add batch dimension (B, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def expand_bounding_box(bbox, image_width, image_height, expansion_factor=0.05):
    """
    Expands a bounding box by a given factor, ensuring it stays within image bounds.
    This can help make smaller detections more visible.
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    expand_x = width * expansion_factor
    expand_y = height * expansion_factor
    
    # Apply expansion and clamp to image boundaries
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(image_width, x2 + expand_x)
    new_y2 = min(image_height, y2 + expand_y)
    
    return new_x1, new_y1, new_x2, new_y2

def process_single_image(image_data, frontend_image_id, filename=None):
    """
    Processes a single image (either file stream or base64) for defect detection.
    Returns detection results including bounding boxes and image dimensions.
    """
    try:
        # Handle different image input types
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Base64 encoded image (from camera or JSON upload)
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # File upload stream
            image = Image.open(image_data.stream).convert('RGB')
        
        original_width, original_height = image.size
        print(f"Processing image: {filename or frontend_image_id}, Original Size: {original_width}x{original_height}")
        
        # Preprocess image for model input
        img_tensor = preprocess_image(image).to(device)
        
        # Perform inference
        with torch.no_grad():
            predictions = model(img_tensor)

        # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
        conf_thres = 0.25  # Confidence threshold for detections
        iou_thres = 0.45   # IoU threshold for NMS
        
        try:
            # Use YOLOv5's NMS function
            pred = non_max_suppression(predictions, conf_thres, iou_thres)
        except Exception as nms_error:
            print(f"NMS error: {nms_error}. Falling back to direct prediction processing.")
            # Fallback if NMS fails (e.g., due to unexpected prediction format)
            pred = predictions
            if isinstance(pred, tuple): # Handle cases where predictions might be a tuple
                pred = pred[0]
            if not isinstance(pred, list): # Ensure it's iterable
                pred = [pred]
        
        detections = []
        
        # Process each detected object
        for i, det in enumerate(pred):  # detections per image in the batch
            if len(det):
                # Rescale bounding boxes from model's input size (640x640) back to original image size
                try:
                    det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], (original_height, original_width)).round()
                except Exception as scale_error:
                    print(f"Scale boxes error: {scale_error}. Performing manual rescaling.")
                    # Manual rescaling if scale_boxes fails (e.g., due to missing utility)
                    scale_x = original_width / img_tensor.shape[3] # img_tensor.shape[3] is width (640)
                    scale_y = original_height / img_tensor.shape[2] # img_tensor.shape[2] is height (640)
                    det[:, 0] *= scale_x  # x1
                    det[:, 1] *= scale_y  # y1
                    det[:, 2] *= scale_x  # x2
                    det[:, 3] *= scale_y  # y2
                
                # Iterate through each individual detection
                for *xyxy, conf, cls in reversed(det):
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    # Get human-readable class name
                    class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
                    
                    # Convert coordinates to float and ensure they are within image bounds
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    
                    x1 = max(0, min(x1, original_width))
                    y1 = max(0, min(y1, original_height))
                    x2 = max(0, min(x2, original_width))
                    y2 = max(0, min(y2, original_height))
                    
                    # Skip invalid bounding boxes (e.g., zero width/height)
                    if x2 <= x1 or y2 <= y1:
                        print(f"  Skipping invalid bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                        continue
                    
                    # Optionally expand bounding box for better visibility on frontend
                    x1_exp, y1_exp, x2_exp, y2_exp = expand_bounding_box(
                        (x1, y1, x2, y2), original_width, original_height
                    )
                    
                    detection_data = {
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": { # Use 'bbox' for consistency with frontend
                            "x1": float(x1_exp),
                            "y1": float(y1_exp),
                            "x2": float(x2_exp),
                            "y2": float(y2_exp)
                        }
                    }
                    
                    detections.append(detection_data)
                    print(f"  Detection: {class_name} at ({x1_exp:.1f}, {y1_exp:.1f}) to ({x2_exp:.1f}, {y2_exp:.1f}) with confidence {confidence:.3f}")

        # Sort detections by confidence and limit the number of results sent to frontend
        max_detections_per_image = 20 # Limit to prevent overwhelming frontend
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_detections_per_image]

        result = {
            "image_id": frontend_image_id, # Use the ID from the frontend for mapping
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
            "image_id": frontend_image_id, # Return ID even on error
            "error": f"Processing failed: {str(e)}",
            "predictions": [],
            "image_dimensions": {"width": 0, "height": 0}, # Indicate unknown dimensions on error
            "total_detections": 0
        }

@app.route('/')
def home():
    """Simple home route to confirm backend is running."""
    return "PCB Defect Detector Backend is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image prediction requests. Supports:
    1. JSON array of base64 images (for camera or mixed types from frontend).
    2. FormData with multiple file uploads and associated metadata.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check backend logs for details."}), 500

    results = []
    total_defects = 0
    processing_errors = []

    print("\n--- Starting prediction request ---")
    
    # Case 1: JSON array of image data (e.g., from camera captures or if frontend sends all as base64)
    if request.is_json and 'images_data' in request.json:
        images_data = request.json['images_data']
        print(f"Received {len(images_data)} images from JSON data.")
        for img_data in images_data:
            frontend_id = img_data.get('id', f"json_{uuid.uuid4().hex[:8]}") # Get frontend ID or generate fallback
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

    # Case 2: FormData with multiple file uploads and associated metadata
    elif 'images' in request.files:
        files = request.files.getlist('images')
        image_metadata_json = request.form.get('image_metadata')
        image_metadata = []
        if image_metadata_json:
            try:
                image_metadata = json.loads(image_metadata_json)
            except json.JSONDecodeError:
                print("Warning: Could not decode image_metadata JSON from frontend. IDs might not match.")

        print(f"Received {len(files)} uploaded files via FormData.")
        
        for i, file in enumerate(files):
            if file and file.filename:
                frontend_id = f"upload_{uuid.uuid4().hex[:8]}" # Default fallback ID
                frontend_name = file.filename
                
                # Try to find matching metadata by index (assuming order is preserved)
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
    
    # Fallback for single image upload (older frontend versions might still use this)
    elif 'image' in request.files:
        file = request.files['image']
        if file and file.filename:
            frontend_id = f"single_{uuid.uuid4().hex[:8]}" # Generate new ID for old format
            print(f"Received single image '{file.filename}' via old format.")
            result = process_single_image(file, frontend_id, file.filename)
            results.append(result)
            if 'error' not in result:
                total_defects += result['total_detections']
            else:
                processing_errors.append(f"Image '{file.filename}' (ID: {frontend_id}): {result['error']}")
    
    if not results:
        return jsonify({"error": "No images provided for processing. Please upload images or use the camera."}), 400
    
    # Generate summary statistics from all processed results
    defect_summary = {}
    for result in results:
        if 'error' not in result: # Only count defects from successfully processed images
            for detection in result['predictions']:
                defect_type = detection['class']
                defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
    
    response_data = {
        "results": results, # Array of results for each image
        "summary": {
            "total_images_processed": len(results),
            "total_defects_found": total_defects,
            "defect_breakdown": defect_summary,
            "processing_errors": processing_errors # List of errors encountered
        },
        "debug_info": { # Useful for debugging backend issues
            "model_device": str(device),
            "model_names": getattr(model, 'names', CLASS_NAMES),
            "total_results": len(results)
        }
    }
    
    # Debug information for backend console
    print(f"--- Prediction complete ---")
    print(f"Processed {len(results)} images.")
    print(f"Total defects found across all images: {total_defects}")
    if defect_summary:
        print("Defect breakdown:")
        for defect_type, count in defect_summary.items():
            print(f"  {defect_type}: {count}")
    if processing_errors:
        print(f"Errors encountered: {len(processing_errors)}")
        for err in processing_errors:
            print(f"  - {err}")
    print("---------------------------\n")
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Run the Flask app
    # Ensure 'last.pt' and 'yolov5' directory are correctly set up.
    # The host '0.0.0.0' makes it accessible from other devices on the network.
    app.run(debug=True, host='0.0.0.0', port=5000) 
