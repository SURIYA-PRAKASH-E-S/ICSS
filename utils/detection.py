import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from .tracker import ObjectTracker

prev_positions = {}
centroid_history = []  # Store movement vectors for flow analysis

# ================= ENHANCED DETECTION PIPELINE =================
def process_frame(frame, model) -> Dict[str, Any]:
    """
    Enhanced frame processing returning structured detection data
    Returns dict with: processed_frame, bounding_boxes, centroids, people_count
    """
    global prev_positions, centroid_history
    
    # Run YOLO inference with verbose=False to suppress console output
    results = model(frame, verbose=False)
    
    people_count = 0
    current_positions = {}
    bounding_boxes = []
    centroids = []
    
    for r in results:
        for i, box in enumerate(r.boxes):
            
            cls = int(box.cls[0])
            
            # Class 0 = person
            if cls == 0:
                people_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bounding_boxes.append((x1, y1, x2, y2))
                
                # Center point (centroid)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                centroids.append((cx, cy))
                current_positions[i] = (cx, cy)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Direction detection (existing logic)
                if i in prev_positions:
                    px, py = prev_positions[i]
                    
                    if cx > px:
                        direction = "Right"
                    elif cx < px:
                        direction = "Left"
                    elif cy > py:
                        direction = "Down"
                    else:
                        direction = "Up"
                    
                    cv2.putText(frame, direction, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
    
    prev_positions = current_positions
    
    return {
        'processed_frame': frame,
        'bounding_boxes': bounding_boxes,
        'centroids': centroids,
        'people_count': people_count
    }

# ================= DUAL MODEL DETECTION PIPELINE =================
def process_frame_dual_models(frame, model_v11, model_v8, active_models=["v11", "v8"]) -> Dict[str, Any]:
    """
    Enhanced frame processing using dual models for improved accuracy
    Returns dict with: processed_frame, bounding_boxes, centroids, people_count, model_info
    """
    global prev_positions, centroid_history
    
    all_detections = []
    model_info = {}
    
    # Process with selected models
    if "v11" in active_models:
        try:
            results_v11 = model_v11(frame, verbose=False)  # Suppress console output
            model_info["v11"] = "Active"
            for r in results_v11:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person
                        all_detections.append({
                            'model': 'v11',
                            'box': box,
                            'confidence': float(box.conf[0])
                        })
        except Exception as e:
            model_info["v11"] = f"Error: {str(e)}"
    
    if "v8" in active_models:
        try:
            results_v8 = model_v8(frame, verbose=False)  # Suppress console output
            model_info["v8"] = "Active"
            for r in results_v8:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person
                        all_detections.append({
                            'model': 'v8',
                            'box': box,
                            'confidence': float(box.conf[0])
                        })
        except Exception as e:
            model_info["v8"] = f"Error: {str(e)}"
    
    # Merge detections (non-maximum suppression style)
    merged_detections = merge_detections(all_detections)
    
    people_count = 0
    current_positions = {}
    bounding_boxes = []
    centroids = []
    
    for detection in merged_detections:
        people_count += 1
        box = detection['box']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bounding_boxes.append((x1, y1, x2, y2))
        
        # Center point (centroid)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroids.append((cx, cy))
        current_positions[people_count-1] = (cx, cy)
        
        # Draw bounding box with model-specific colors
        color = (0, 255, 0) if detection['model'] == 'v11' else (255, 0, 0)  # Green for v11, Blue for v8
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add model label and confidence
        label = f"{detection['model'].upper()}: {detection['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Direction detection (existing logic)
        if (people_count-1) in prev_positions:
            px, py = prev_positions[people_count-1]
            
            if cx > px:
                direction = "Right"
            elif cx < px:
                direction = "Left"
            elif cy > py:
                direction = "Down"
            else:
                direction = "Up"
            
            cv2.putText(frame, direction, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
    
    prev_positions = current_positions
    
    return {
        'processed_frame': frame,
        'bounding_boxes': bounding_boxes,
        'centroids': centroids,
        'people_count': people_count,
        'model_info': model_info,
        'detection_count': len(all_detections),
        'merged_count': len(merged_detections)
    }

def merge_detections(detections, iou_threshold=0.5):
    """
    Merge detections from multiple models using IoU-based NMS
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    merged = []
    for detection in detections:
        box = detection['box']
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        
        keep = True
        for merged_det in merged:
            merged_box = merged_det['box']
            mx1, my1, mx2, my2 = map(float, merged_box.xyxy[0])
            
            # Calculate IoU
            xi1 = max(x1, mx1)
            yi1 = max(y1, my1)
            xi2 = min(x2, mx2)
            yi2 = min(y2, my2)
            
            if xi2 > xi1 and yi2 > yi1:
                intersection = (xi2 - xi1) * (yi2 - yi1)
                union = ((x2 - x1) * (y2 - y1)) + ((mx2 - mx1) * (my2 - my1)) - intersection
                iou = intersection / union
                
                if iou > iou_threshold:
                    keep = False
                    break
        
        if keep:
            merged.append(detection)
    
    return merged

# ================= ADVANCED CROWD TRACKING =================
def track_movement(current_centroids: List[Tuple[int, int]], 
                  prev_centroids: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Track movement between frames using centroid matching
    Returns movement vectors (dx, dy) for matched centroids
    """
    movement_vectors = []
    
    if not prev_centroids or not current_centroids:
        return movement_vectors
    
    # Simple nearest centroid matching
    matched_pairs = []
    used_indices = set()
    
    for i, (curr_cx, curr_cy) in enumerate(current_centroids):
        min_dist = float('inf')
        best_match = -1
        
        for j, (prev_cx, prev_cy) in enumerate(prev_centroids):
            if j in used_indices:
                continue
                
            dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            if dist < min_dist and dist < 100:  # Max distance threshold
                min_dist = dist
                best_match = j
        
        if best_match != -1:
            matched_pairs.append((i, best_match))
            used_indices.add(best_match)
    
    # Calculate movement vectors
    for curr_idx, prev_idx in matched_pairs:
        curr_cx, curr_cy = current_centroids[curr_idx]
        prev_cx, prev_cy = prev_centroids[prev_idx]
        
        dx = curr_cx - prev_cx
        dy = curr_cy - prev_cy
        movement_vectors.append((dx, dy))
    
    return movement_vectors

# ================= FLOW DIRECTION INTELLIGENCE =================
def calculate_flow_direction(movement_vectors: List[Tuple[int, int]]) -> str:
    """
    Calculate overall crowd flow direction using majority voting
    Returns: 'Left', 'Right', 'Up', 'Down', or 'Mixed'
    """
    if not movement_vectors:
        return "Unknown"
    
    directions = []
    
    for dx, dy in movement_vectors:
        # Determine dominant direction for each vector
        if abs(dx) > abs(dy):  # Horizontal movement dominant
            if dx > 0:
                directions.append("Right")
            else:
                directions.append("Left")
        else:  # Vertical movement dominant
            if dy > 0:
                directions.append("Down")
            else:
                directions.append("Up")
    
    if not directions:
        return "Unknown"
    
    # Majority voting
    from collections import Counter
    direction_counts = Counter(directions)
    most_common = direction_counts.most_common(1)[0][0]
    
    # Check if it's truly dominant or mixed
    if direction_counts[most_common] / len(directions) > 0.6:
        return most_common
    else:
        return "Mixed"

# ================= IMPROVED DENSITY CALCULATION =================
def calculate_density(people_count: int, frame_width: int, frame_height: int) -> float:
    """
    Calculate normalized density as people per pixel area
    """
    if frame_width <= 0 or frame_height <= 0:
        return 0.0
    
    pixel_area = frame_width * frame_height
    density = people_count / pixel_area
    
    # Normalize for better visualization (multiply by 10000 for readability)
    normalized_density = density * 10000
    
    return normalized_density

# ================= RISK CLASSIFICATION SYSTEM =================
def classify_risk(density: float, people_count: int,
                 low_count_threshold: int = 5, medium_count_threshold: int = 15,
                 low_density_threshold: float = 0.3, medium_density_threshold: float = 0.7) -> Tuple[str, Tuple[int, int, int]]:
    """
    3-level risk classification system (EXACT LOGIC)
    Returns: (risk_level, color_tuple)
    """
    if people_count >= medium_count_threshold or density >= medium_density_threshold:
        return "HIGH RISK", (0, 0, 255)  # Red
    elif people_count >= low_count_threshold or density >= low_density_threshold:
        return "MEDIUM RISK", (0, 165, 255)  # Orange
    else:
        return "LOW RISK", (0, 255, 0)  # Green

# ================= LEGACY FUNCTION (for backward compatibility) =================
def process_frame_legacy(frame, model):
    """
    Legacy process_frame function for backward compatibility
    """
    result = process_frame(frame, model)
    
    # Add legacy overlays
    h, w, _ = frame.shape
    density = calculate_density(result['people_count'], w, h)
    risk_level, risk_color = classify_risk(density, result['people_count'])
    
    # Risk Alert
    if risk_level == "Risky":
        cv2.putText(frame, " High Risk",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, risk_color, 3)
    
    # Count display
    cv2.putText(frame, f"Count: {result['people_count']}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)
    
    return frame

# ================= DEEP SORT INTEGRATED TRACKING =================
def process_frame_with_deep_sort(frame, model_v11, model_v8, tracker: ObjectTracker, active_models=["v11", "v8"], fps=30, model_v11m=None) -> Dict[str, Any]:
    """
    Process frame with YOLO detection + Deep SORT tracking
    
    Args:
        frame: Input video frame
        model_v11: YOLO v11 model
        model_v8: YOLO v8 model  
        model_v11m: YOLO v11m model (optional)
        tracker: Deep SORT tracker instance
        active_models: List of active models ["v11", "v8", "v11m"]
        fps: Frames per second for speed calculation
        
    Returns:
        Dictionary with tracking results and annotated frame
    """
    try:
        # Update tracker FPS
        tracker.update_fps(fps)
        
        # Get YOLO detections from active models
        all_detections = []
        confidences = []
        model_info = {}
        
        # Process with selected models
        if "v11" in active_models:
            try:
                results_v11 = model_v11(frame, verbose=False)  # Suppress console output
                model_info["v11"] = "Active"
                for r in results_v11:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_detections.append((x1, y1, x2, y2))
                            confidences.append(float(box.conf[0]))
            except Exception as e:
                model_info["v11"] = f"Error: {str(e)}"
        
        if "v8" in active_models:
            try:
                results_v8 = model_v8(frame, verbose=False)  # Suppress console output
                model_info["v8"] = "Active"
                for r in results_v8:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_detections.append((x1, y1, x2, y2))
                            confidences.append(float(box.conf[0]))
            except Exception as e:
                model_info["v8"] = f"Error: {str(e)}"
        
        if "v11m" in active_models and model_v11m is not None:
            try:
                results_v11m = model_v11m(frame, verbose=False)  # Suppress console output
                model_info["v11m"] = "Active"
                for r in results_v11m:
                    for box in r.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_detections.append((x1, y1, x2, y2))
                            confidences.append(float(box.conf[0]))
            except Exception as e:
                model_info["v11m"] = f"Error: {str(e)}"
        
        # Handle case with no detections
        if not all_detections:
            # Return frame with basic info when no detections
            h, w, _ = frame.shape
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No persons detected", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return {
                'annotated_frame': annotated_frame,
                'tracking_results': {'tracked_objects': [], 'total_count': 0, 'track_history': {}},
                'people_count': 0,
                'density': 0.0,
                'flow_direction': 'Unknown',
                'risk_level': 'Normal',
                'risk_color': (0, 255, 0),
                'model_info': model_info,
                'avg_speed': 0.0,
                'tracked_objects': []
            }
        
        # Update Deep SORT tracker with error handling
        try:
            # Ensure frame is in proper format for Deep SORT (BGR)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Frame is already in correct format
                tracker_frame = frame
            else:
                # Convert frame to BGR if needed
                tracker_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 4 else frame
            
            tracking_results = tracker.update_tracks(tracker_frame, all_detections, confidences)
        except Exception as e:
            print(f"Deep SORT tracking error: {e}")
            # Fallback to basic detection without tracking
            h, w, _ = frame.shape
            annotated_frame = frame.copy()
            
            # Draw basic detection boxes
            for (x1, y1, x2, y2), conf in zip(all_detections, confidences):
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Person {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return {
                'annotated_frame': annotated_frame,
                'tracking_results': {'tracked_objects': [], 'total_count': len(all_detections), 'track_history': {}},
                'people_count': len(all_detections),
                'density': calculate_density(len(all_detections), w, h),
                'flow_direction': 'Unknown',
                'risk_level': 'Normal',
                'risk_color': (0, 255, 0),
                'model_info': model_info,
                'avg_speed': 0.0,
                'tracked_objects': []
            }
        
        # Draw tracking information on frame with error handling
        try:
            annotated_frame = tracker.draw_tracking_info(frame, tracking_results)
            
            # ISSUE 1 FIX: If tracker returns 0 tracks, fall back to drawing raw YOLO detections
            if len(tracking_results.get('tracked_objects', [])) == 0 and all_detections:
                for (x1, y1, x2, y2), conf in zip(all_detections, confidences):
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Person {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Drawing tracking info error: {e}")
            annotated_frame = frame.copy()
        
        # ISSUE 2 FIX: Add debug diagnostics
        print(f"[DEBUG] YOLO detections: {len(all_detections)}, Deep SORT tracks: {len(tracking_results.get('tracked_objects', []))}")
        if tracking_results.get('tracked_objects'):
            track_ids = [obj['track_id'] for obj in tracking_results['tracked_objects']]
            print(f"[DEBUG] Active track IDs: {track_ids}")
        
        # Calculate additional metrics
        h, w, _ = frame.shape
        people_count = tracking_results['total_count']
        density = calculate_density(people_count, w, h)
        
        # Extract tracked objects data for analytics
        tracked_objects = tracking_results.get('tracked_objects', [])
        
        # Calculate flow direction from tracked objects
        try:
            directions = [obj['direction'] for obj in tracked_objects if obj.get('direction') != 'Unknown']
            flow_dir = calculate_flow_direction_from_directions(directions)
        except Exception as e:
            print(f"Flow direction calculation error: {e}")
            flow_dir = 'Unknown'
        
        # Calculate average speed
        try:
            speeds = [obj['speed'] for obj in tracked_objects if obj.get('speed', 0) > 0]
            avg_speed = np.mean(speeds) if speeds else 0.0
        except Exception as e:
            print(f"Average speed calculation error: {e}")
            avg_speed = 0.0
        
        # Classify risk
        try:
            risk_level, risk_color = classify_risk(
                density, 
                people_count,
                low_count_threshold=5,
                medium_count_threshold=15,
                low_density_threshold=0.3,
                medium_density_threshold=0.7
            )
        except Exception as e:
            print(f"Risk classification error: {e}")
            risk_level, risk_color = 'LOW RISK', (0, 255, 0)
        
        return {
            'annotated_frame': annotated_frame,
            'tracking_results': tracking_results,
            'people_count': people_count,
            'density': density,
            'flow_direction': flow_dir,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_info': model_info,
            'avg_speed': avg_speed,
            'tracked_objects': tracked_objects
        }
        
    except Exception as e:
        print(f"Critical error in process_frame_with_deep_sort: {e}")
        # Return safe fallback
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, f"Tracking Error: {str(e)}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return {
            'annotated_frame': annotated_frame,
            'tracking_results': {'tracked_objects': [], 'total_count': 0, 'track_history': {}},
            'people_count': 0,
            'density': 0.0,
            'flow_direction': 'Unknown',
            'risk_level': 'Normal',
            'risk_color': (0, 255, 0),
            'model_info': {'error': str(e)},
            'avg_speed': 0.0,
            'tracked_objects': []
        }

def calculate_flow_direction_from_directions(directions: List[str]) -> str:
    """
    Calculate overall flow direction from individual object directions
    """
    if not directions:
        return "Unknown"
    
    # Count directions
    direction_counts = {
        'North': 0, 'South': 0, 'East': 0, 'West': 0
    }
    
    for direction in directions:
        if direction in direction_counts:
            direction_counts[direction] += 1
    
    # Find dominant direction
    max_count = max(direction_counts.values())
    if max_count == 0:
        return "Unknown"
    
    dominant_directions = [d for d, count in direction_counts.items() if count == max_count]
    
    if len(dominant_directions) == 1:
        return dominant_directions[0]
    else:
        # If tie, check for opposing directions
        if 'North' in dominant_directions and 'South' in dominant_directions:
            return "Mixed"
        elif 'East' in dominant_directions and 'West' in dominant_directions:
            return "Mixed"
        else:
            return dominant_directions[0]