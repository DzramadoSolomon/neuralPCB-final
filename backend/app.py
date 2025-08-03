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
import json # Import json for handling metadata
import uuid # Import uuid for generating unique IDs

# Import YOLO from ultralytics
# Ensure 'ultralytics' is in your requirements.txt
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error importing ultralytics: {e}")
    print("Please ensure ultralytics is installed. Run: pip install ultralytics")
    sys.exit(1) # Exit if YOLO cannot be imported

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for frontend-backend communication.
# IMPORTANT: In a production environment, replace "*" with your Vercel frontend domain (e.g., "https://your-vercel-app.vercel.app")
CORS(app)

# Determine device for model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model
# Make sure 'best.pt' or 'last.pt' is in the same directory as app.py
try:
    # Attempt to load best.pt first, fallback to last.pt if not found
    if os.path.exists('best.pt'):
        model = YOLO('best.pt')
        print("Loaded model: best.pt")
    elif os.path.exists('last.pt'):
        model = YOLO('last.pt')
        print("Loaded model: last.pt")
    else:
        raise FileNotFoundError("Neither best.pt nor last.pt found in the current directory.")
    
    # Ensure model is on the correct device
    model.to(device)
    print(f"Model moved to {device}")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None to indicate failure
    # Fallback/error handling for model loading
    # In a real application, you might want to stop the server or serve a maintenance page
    # For now, we'll allow the app to run but predictions will fail.

# Define class names mapping from the model
# Ultralytics models have a .names attribute
CLASS_NAMES = model.names if model and hasattr(model, 'names') else {
    0: 'missing_hole',
    1: 'mouse_bite',
    2: 'open_circuit',
    3: 'short',
    4: 'spur',
    5: 'spurious_copper'
}
print(f"Model class names: {CLASS_NAMES}")

@app.route('/')
def home():
    """Simple home route for health check."""
    return "PCB Defect Detector Backend is running!", 200

# This function is no longer needed with ultralytics' direct prediction
# def preprocess_image(image, target_size=(640, 640)):
#     """
#     Preprocesses a PIL Image for YOLOv5 inference.
#     Resizes the image and normalizes pixel values.
#     """
#     image_resized = image.resize(target_size)
#     img_array = np.array(image_resized)
#     img_tensor = torch.from_numpy(img_array).float() / 255.0
#     if img_tensor.ndimension() == 3:
#         img_tensor = img_tensor.permute(2, 0, 1)
#     img_tensor = img_tensor.unsqueeze(0)
#     return img_tensor

# This function is for optional visual expansion, not core to model accuracy
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
        
        # Perform inference using the ultralytics model
        # model.predict can take PIL Image directly.
        # verbose=False suppresses logging from ultralytics.
        # imgsz=640 ensures input size consistency.
        preds = model.predict(image, verbose=False, imgsz=640) 
        
        detections = []
        
        # Process each detected object from the Results object
        for pred in preds: # preds is a list of Results objects, one per image
            if pred.boxes: # Check if bounding boxes are detected
                for box in pred.boxes:
                    # box.xyxy: [x1, y1, x2, y2] in pixels (already scaled to original image size by ultralytics)
                    # box.conf: confidence score
                    # box.cls: class ID
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')

                    # Optionally apply expansion for better visibility on frontend
                    # If you want exact boxes, remove the expand_bounding_box call
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
        return jsonify({"error": "No valid images provided for detection."}), 400
    
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
            "model_names": CLASS_NAMES,
            "total_results": len(results)
        }
    }
    
    print(f"Processed {len(results)} images. Total defects found: {total_defects}")
    return jsonify(response_data), 200

if __name__ == '__main__':
    # Run the Flask app
    # Ensure 'last.pt' and 'yolov5' directory are correctly set up.
    # The host '0.0.0.0' makes it accessible from other devices on the network.
    app.run(debug=True, host='0.0.0.0', port=5000)
