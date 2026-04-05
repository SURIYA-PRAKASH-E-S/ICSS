"""
Deep SORT Multi-Object Tracking Module
Integrates with YOLO detections for persistent object tracking
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple, Dict, Any
from collections import deque
import time

class ObjectTracker:
    """
    Deep SORT based multi-object tracker with speed and direction estimation
    """
    
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=0.3):
        """
        Initialize Deep SORT tracker
        
        Args:
            max_age: Maximum number of frames to keep track without updates
            n_init: Number of consecutive detections before track is confirmed
            nms_max_overlap: NMS threshold for detection overlap
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap
        )
        
        # Track history for each object (last N positions)
        self.track_history = {}
        self.max_history = 20  # Keep last 20 positions
        
        # Speed estimation parameters
        self.speed_history = {}  # Moving average for speed smoothing
        self.speed_window = 5  # Window for moving average
        
        # FPS for speed calculation
        self.fps = 30  # Default, will be updated
        self.last_frame_time = time.time()
        
    def update_fps(self, fps):
        """Update FPS for accurate speed calculation"""
        self.fps = fps if fps > 0 else 30
        
    def detections_to_deepsort_format(self, detections: List[Tuple], confidences: List[float]) -> List[Tuple]:
        """
        Convert YOLO detections to Deep SORT format
        
        Args:
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            confidences: List of confidence scores
            
        Returns:
            List of detections in Deep SORT format [([x, y, w, h], confidence, class), ...]
        """
        deepsort_detections = []
        
        for (x1, y1, x2, y2), conf in zip(detections, confidences):
            # Convert to (x, y, w, h) format
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            deepsort_detections.append(([x, y, w, h], conf, 'person'))
            
        return deepsort_detections
    
    def calculate_speed(self, track_id: int, current_pos: Tuple[float, float]) -> float:
        """
        Calculate speed for tracked object using pixel displacement
        
        Args:
            track_id: Unique track identifier
            current_pos: Current position (x, y)
            
        Returns:
            Speed in pixels per second
        """
        current_time = time.time()
        
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            self.track_history[track_id] = deque(maxlen=self.max_history)
            self.track_history[track_id].append(current_pos)
            self.speed_history[track_id] = deque(maxlen=self.speed_window)
            return 0.0
        
        # Get previous position
        prev_pos = self.track_history[track_id][-1]
        
        # Calculate pixel distance
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate time delta
        time_delta = 1.0 / self.fps  # Use FPS for consistent timing
        
        # Calculate speed (pixels per second)
        if time_delta > 0:
            speed = distance / time_delta
        else:
            speed = 0.0
        
        # Apply smoothing with moving average
        self.speed_history[track_id].append(speed)
        
        if len(self.speed_history[track_id]) > 0:
            smoothed_speed = np.mean(self.speed_history[track_id])
        else:
            smoothed_speed = speed
            
        # Update position history
        self.track_history[track_id].append(current_pos)
        
        return smoothed_speed
    
    def calculate_direction(self, track_id: int, current_pos: Tuple[float, float]) -> str:
        """
        Calculate movement direction based on displacement
        
        Args:
            track_id: Unique track identifier
            current_pos: Current position (x, y)
            
        Returns:
            Direction string: 'North', 'South', 'East', 'West', or 'Unknown'
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 'Unknown'
        
        # Get previous position
        prev_pos = self.track_history[track_id][-2]  # Use second-to-last for better accuracy
        
        # Calculate displacement
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Apply threshold to avoid noise (minimum movement in pixels)
        threshold = 5.0
        if abs(dx) < threshold and abs(dy) < threshold:
            return 'Unknown'
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal movement is dominant
            return 'East' if dx > 0 else 'West'
        else:
            # Vertical movement is dominant
            return 'South' if dy > 0 else 'North'
    
    def update_tracks(self, frame: np.ndarray, detections: List[Tuple], confidences: List[float]) -> Dict[str, Any]:
        """
        Update tracker with new detections
        
        Args:
            frame: Current video frame
            detections: List of bounding boxes [(x1, y1, x2, y2), ...]
            confidences: List of confidence scores
            
        Returns:
            Dictionary containing tracking results
        """
        try:
            # Convert detections to Deep SORT format
            deepsort_detections = self.detections_to_deepsort_format(detections, confidences)
            
            # Validate frame format and dimensions
            if frame is None or frame.size == 0:
                return {
                    'tracked_objects': [],
                    'total_count': len(detections),
                    'track_history': dict(self.track_history)
                }
            
            # Ensure frame is in correct format (BGR, 3 channels)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Frame is already in correct format
                processed_frame = frame
            else:
                # Convert frame to correct format
                if len(frame.shape) == 2:
                    # Grayscale to BGR
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    # Ensure 3 channels
                    processed_frame = frame
            
            # Update tracker with error handling
            try:
                tracks = self.tracker.update_tracks(deepsort_detections, processed_frame)
            except Exception as e:
                # Suppress error message for cleaner output
                # print(f"Deep SORT tracker update error: {e}")
                # Return empty tracking results on tracker error
                return {
                    'tracked_objects': [],
                    'total_count': len(detections),  # Still return detection count
                    'track_history': dict(self.track_history)
                }
            
            # Process tracks with error handling
            tracked_objects = []
            total_count = len(tracks)
            
            for track in tracks:
                try:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    
                    # Get bounding box with error handling
                    try:
                        bbox = track.to_ltwh()  # (x, y, w, h)
                        # Ensure bbox is a tuple of integers
                        bbox = tuple(int(coord) if not isinstance(coord, str) else 0 for coord in bbox)
                    except Exception as e:
                        # Suppress error message
                        # print(f"Error getting bbox for track {track_id}: {e}")
                        continue
                    
                    # Validate bounding box
                    if len(bbox) != 4 or any(coord < 0 for coord in bbox):
                        continue
                    
                    # Calculate center position with proper type conversion
                    try:
                        center_x = float(bbox[0]) + float(bbox[2]) / 2
                        center_y = float(bbox[1]) + float(bbox[3]) / 2
                        current_pos = (int(center_x), int(center_y))
                    except (ValueError, TypeError):
                        continue
                    
                    # Calculate speed and direction with error handling
                    try:
                        speed = self.calculate_speed(track_id, current_pos)
                    except Exception as e:
                        # Suppress error message
                        # print(f"Error calculating speed for track {track_id}: {e}")
                        speed = 0.0
                    
                    try:
                        direction = self.calculate_direction(track_id, current_pos)
                    except Exception as e:
                        # Suppress error message
                        # print(f"Error calculating direction for track {track_id}: {e}")
                        direction = 'Unknown'
                    
                    # Update track history with proper type conversion
                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=self.max_history)
                    self.track_history[track_id].append(current_pos)
                    
                    # Create tracked object with proper types
                    tracked_object = {
                        'track_id': int(track_id),
                        'bbox': bbox,  # Already converted to integers
                        'center': current_pos,  # Already converted to integers
                        'speed': float(speed),
                        'direction': str(direction),
                        'confidence': float(confidences[0]) if confidences else 0.0
                    }
                    
                    tracked_objects.append(tracked_object)
                    
                except Exception as e:
                    # Suppress error message
                    # print(f"Error processing track: {e}")
                    continue
            
            return {
                'tracked_objects': tracked_objects,
                'total_count': total_count,
                'track_history': dict(self.track_history)
            }
            
        except Exception as e:
            # Suppress error message
            # print(f"Critical error in update_tracks: {e}")
            # Return safe fallback
            return {
                'tracked_objects': [],
                'total_count': 0,
                'track_history': dict(self.track_history)
            }
    
    def draw_tracking_info(self, frame: np.ndarray, tracking_results: Dict[str, Any]) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            tracking_results: Results from update_tracks()
            
        Returns:
            Frame with tracking overlays
        """
        annotated_frame = frame.copy()
        
        for obj in tracking_results['tracked_objects']:
            try:
                track_id = obj['track_id']
                bbox = obj['bbox']
                center = obj['center']
                speed = obj['speed']
                direction = obj['direction']
                
                # Ensure all values are proper types
                track_id = int(track_id) if not isinstance(track_id, str) else 0
                bbox = tuple(int(coord) if not isinstance(coord, str) else 0 for coord in bbox)
                center = tuple(int(coord) if not isinstance(coord, str) else 0 for coord in center)
                speed = float(speed) if not isinstance(speed, str) else 0.0
                direction = str(direction)
                
                # Draw bounding box with unique color per ID
                color = self._get_color_for_id(track_id)
                
                # Ensure coordinates are integers for OpenCV functions
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                
                # Draw rectangle with proper integer coordinates
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point with proper integer coordinates
                cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 3, color, -1)
                
                # Draw trajectory line (last N positions)
                if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    positions = list(self.track_history[track_id])
                    for i in range(1, len(positions)):
                        try:
                            # Ensure positions are integers
                            prev_pos = (int(positions[i-1][0]), int(positions[i-1][1]))
                            curr_pos = (int(positions[i][0]), int(positions[i][1]))
                            cv2.line(annotated_frame, prev_pos, curr_pos, color, 2)
                        except (ValueError, TypeError, IndexError):
                            continue
                
                # Draw ID and info text
                info_text = f"ID: {track_id} | Speed: {speed:.1f} px/s | {direction}"
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for text with proper integer coordinates
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - text_size[1] - 10),
                             (x1 + text_size[0], y1 - 5),
                             color, -1)
                
                # Text with proper integer coordinates
                cv2.putText(annotated_frame, info_text,
                            (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw direction arrow if moving
                if direction != 'Unknown' and track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    try:
                        self._draw_direction_arrow(annotated_frame, (int(center[0]), int(center[1])), direction, color)
                    except Exception:
                        continue
                        
            except Exception as e:
                # Suppress drawing errors for cleaner output
                # print(f"Drawing tracking info error: {e}")
                continue
        
        return annotated_frame
    
    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def _draw_direction_arrow(self, frame: np.ndarray, center: Tuple[int, int], direction: str, color: Tuple[int, int, int]):
        """Draw direction arrow at object center"""
        arrow_length = 30
        arrow_end = center
        
        if direction == 'North':
            arrow_end = (center[0], center[1] - arrow_length)
        elif direction == 'South':
            arrow_end = (center[0], center[1] + arrow_length)
        elif direction == 'East':
            arrow_end = (center[0] + arrow_length, center[1])
        elif direction == 'West':
            arrow_end = (center[0] - arrow_length, center[1])
        
        cv2.arrowedLine(frame, center, arrow_end, color, 3, tipLength=0.3)

def init_tracker(max_age=30, n_init=3, nms_max_overlap=0.3) -> ObjectTracker:
    """
    Initialize and return ObjectTracker instance
    
    Args:
        max_age: Maximum number of frames to keep track without updates
        n_init: Number of consecutive detections before track is confirmed
        nms_max_overlap: NMS threshold for detection overlap
        
    Returns:
        Initialized ObjectTracker instance
    """
    return ObjectTracker(max_age, n_init, nms_max_overlap)
