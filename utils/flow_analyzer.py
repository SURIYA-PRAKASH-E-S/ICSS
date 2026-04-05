"""
Flow Analysis Engine for Intelligent Crowd Analytics
Advanced motion pattern detection and crowd flow analysis
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class MotionVector:
    """Represents motion vector for a tracked object."""
    dx: float  # Change in x
    dy: float  # Change in y
    magnitude: float  # Vector magnitude
    angle: float  # Angle in degrees
    direction: str  # Direction category

@dataclass
class FlowMetrics:
    """Flow analysis metrics for current frame."""
    total_objects: int
    direction_distribution: Dict[str, int]
    direction_percentages: Dict[str, float]
    bidirectional_conflict: bool
    surge_detected: bool
    dominant_direction: str
    flow_conflict_score: float
    speed_variation_score: float

class FlowAnalyzer:
    """
    Advanced flow analysis engine for crowd behavior detection.
    Analyzes motion patterns, detects conflicts, and identifies crowd surges.
    """
    
    def __init__(self, 
                 history_length: int = 10,
                 min_movement_threshold: float = 5.0,
                 conflict_threshold: float = 0.3,
                 surge_threshold: float = 0.6,
                 speed_variation_window: int = 5):
        """
        Initialize Flow Analyzer with detection parameters.
        
        Args:
            history_length: Length of position history for tracking
            min_movement_threshold: Minimum movement to consider as motion
            conflict_threshold: Threshold for bidirectional conflict detection
            surge_threshold: Threshold for crowd surge detection
            speed_variation_window: Window for speed variation calculation
        """
        self.history_length = history_length
        self.min_movement_threshold = min_movement_threshold
        self.conflict_threshold = conflict_threshold
        self.surge_threshold = surge_threshold
        self.speed_variation_window = speed_variation_window
        
        # Position history for each tracked object
        self.position_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Speed history for variation calculation
        self.speed_history = defaultdict(lambda: deque(maxlen=speed_variation_window))
        
        # Direction categories
        self.directions = ['Left', 'Right', 'Up', 'Down', 'Up-Left', 'Up-Right', 'Down-Left', 'Down-Right', 'Static']
        
        # Previous frame metrics for surge detection
        self.prev_direction_distribution = defaultdict(int)
        self.prev_total_objects = 0
        
        # Flow state tracking
        self.bidirectional_conflict_active = False
        self.surge_active = False
        self.surge_direction = None
    
    def classify_direction(self, dx: float, dy: float, magnitude: float) -> str:
        """
        Classify motion direction based on dx, dy components.
        
        Args:
            dx: Change in x coordinate
            dy: Change in y coordinate
            magnitude: Movement magnitude
            
        Returns:
            Direction category string
        """
        if magnitude < self.min_movement_threshold:
            return 'Static'
        
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(-dy, dx))  # Negative dy for correct orientation
        
        # Normalize angle to [0, 360]
        if angle < 0:
            angle += 360
        
        # Classify direction based on angle
        if 337.5 <= angle or angle < 22.5:
            return 'Right'
        elif 22.5 <= angle < 67.5:
            return 'Up-Right'
        elif 67.5 <= angle < 112.5:
            return 'Up'
        elif 112.5 <= angle < 157.5:
            return 'Up-Left'
        elif 157.5 <= angle < 202.5:
            return 'Left'
        elif 202.5 <= angle < 247.5:
            return 'Down-Left'
        elif 247.5 <= angle < 292.5:
            return 'Down'
        else:  # 292.5 <= angle < 337.5
            return 'Down-Right'
    
    def compute_motion_vector(self, 
                             track_id: int, 
                             current_x: float, 
                             current_y: float) -> Optional[MotionVector]:
        """
        Compute motion vector for tracked object.
        
        Args:
            track_id: Tracking ID
            current_x: Current x coordinate
            current_y: Current y coordinate
            
        Returns:
            MotionVector object or None if insufficient history
        """
        history = self.position_history[track_id]
        
        # Need at least 2 positions for motion calculation
        if len(history) < 2:
            return None
        
        # Get previous position
        prev_x, prev_y = history[-2]
        
        # Calculate displacement
        dx = current_x - prev_x
        dy = current_y - prev_y
        magnitude = math.sqrt(dx**2 + dy**2)
        
        # Calculate angle
        angle = math.degrees(math.atan2(dy, dx)) if magnitude > 0 else 0
        
        # Classify direction
        direction = self.classify_direction(dx, dy, magnitude)
        
        return MotionVector(dx, dy, magnitude, angle, direction)
    
    def update_tracking_data(self, tracking_results: List[Dict]):
        """
        Update position history with new tracking data.
        
        Args:
            tracking_results: List of tracking results with track_id and position
        """
        # Update position history
        for result in tracking_results:
            track_id = result.get('track_id')
            bbox = result.get('bbox')
            
            if track_id is not None and bbox:
                # Calculate center of bounding box
                x = (bbox[0] + bbox[2]) / 2
                y = (bbox[1] + bbox[3]) / 2
                
                # Add to history
                self.position_history[track_id].append((x, y))
                
                # Calculate and store speed
                motion_vector = self.compute_motion_vector(track_id, x, y)
                if motion_vector:
                    self.speed_history[track_id].append(motion_vector.magnitude)
        
        # Clean up old tracks (remove tracks not seen recently)
        current_track_ids = {result.get('track_id') for result in tracking_results}
        self._cleanup_old_tracks(current_track_ids)
    
    def _cleanup_old_tracks(self, current_track_ids: set):
        """Remove tracks that are no longer active."""
        tracks_to_remove = []
        
        for track_id in self.position_history:
            if track_id not in current_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.position_history[track_id]
            if track_id in self.speed_history:
                del self.speed_history[track_id]
    
    def analyze_direction_distribution(self, motion_vectors: List[MotionVector]) -> Dict[str, int]:
        """
        Analyze distribution of motion directions.
        
        Args:
            motion_vectors: List of motion vectors
            
        Returns:
            Dictionary with direction counts
        """
        distribution = defaultdict(int)
        
        for vector in motion_vectors:
            if vector.magnitude >= self.min_movement_threshold:
                distribution[vector.direction] += 1
        
        return dict(distribution)
    
    def detect_bidirectional_conflict(self, direction_distribution: Dict[str, int]) -> Tuple[bool, float]:
        """
        Detect bidirectional flow conflicts.
        
        Args:
            direction_distribution: Distribution of motion directions
            
        Returns:
            Tuple of (conflict_detected, conflict_score)
        """
        if not direction_distribution:
            return False, 0.0
        
        total_moving = sum(direction_distribution.values())
        if total_moving < 2:
            return False, 0.0
        
        # Define opposing direction pairs
        opposite_pairs = [
            ('Left', 'Right'),
            ('Up', 'Down'),
            ('Up-Left', 'Down-Right'),
            ('Up-Right', 'Down-Left')
        ]
        
        max_conflict_score = 0.0
        
        for dir1, dir2 in opposite_pairs:
            count1 = direction_distribution.get(dir1, 0)
            count2 = direction_distribution.get(dir2, 0)
            
            if count1 > 0 and count2 > 0:
                # Conflict score based on balance of opposing flows
                min_count = min(count1, count2)
                conflict_score = (2 * min_count) / total_moving
                max_conflict_score = max(max_conflict_score, conflict_score)
        
        conflict_detected = max_conflict_score >= self.conflict_threshold
        return conflict_detected, max_conflict_score
    
    def detect_crowd_surge(self, 
                          current_distribution: Dict[str, int], 
                          current_total: int) -> Tuple[bool, Optional[str]]:
        """
        Detect sudden crowd surge in any direction.
        
        Args:
            current_distribution: Current direction distribution
            current_total: Current total moving objects
            
        Returns:
            Tuple of (surge_detected, surge_direction)
        """
        if current_total < 3:
            return False, None
        
        surge_detected = False
        surge_direction = None
        
        # Check for significant increase in any direction
        for direction, current_count in current_distribution.items():
            prev_count = self.prev_direction_distribution.get(direction, 0)
            
            if current_count >= 3:  # Minimum threshold for surge
                # Calculate percentage increase
                if prev_count == 0:
                    increase_ratio = float('inf') if current_count > 0 else 0
                else:
                    increase_ratio = current_count / prev_count
                
                # Detect surge based on ratio and absolute increase
                if (increase_ratio >= 2.0 and current_count - prev_count >= 2) or \
                   (current_count >= 5 and increase_ratio >= 1.5):
                    surge_detected = True
                    surge_direction = direction
                    break
        
        return surge_detected, surge_direction
    
    def calculate_speed_variation(self) -> float:
        """
        Calculate speed variation across all tracked objects.
        
        Returns:
            Speed variation score [0, 1]
        """
        if not self.speed_history:
            return 0.0
        
        all_speeds = []
        for track_id, speed_deque in self.speed_history.items():
            if len(speed_deque) >= 2:
                # Use recent speeds for variation calculation
                recent_speeds = list(speed_deque)[-min(3, len(speed_deque)):]
                all_speeds.extend(recent_speeds)
        
        if len(all_speeds) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_speed = np.mean(all_speeds)
        std_speed = np.std(all_speeds)
        
        if mean_speed == 0:
            return 0.0
        
        # Normalize variation score
        cv = std_speed / mean_speed
        variation_score = np.clip(cv / 2.0, 0.0, 1.0)  # Normalize to [0, 1]
        
        return variation_score
    
    def analyze_flow(self, tracking_results: List[Dict]) -> FlowMetrics:
        """
        Perform comprehensive flow analysis.
        
        Args:
            tracking_results: List of tracking results
            
        Returns:
            FlowMetrics object with analysis results
        """
        # Update tracking data
        self.update_tracking_data(tracking_results)
        
        # Compute motion vectors for all tracked objects
        motion_vectors = []
        for result in tracking_results:
            track_id = result.get('track_id')
            bbox = result.get('bbox')
            
            if track_id is not None and bbox:
                current_x = (bbox[0] + bbox[2]) / 2
                current_y = (bbox[1] + bbox[3]) / 2
                
                vector = self.compute_motion_vector(track_id, current_x, current_y)
                if vector:
                    motion_vectors.append(vector)
        
        # Analyze direction distribution
        direction_distribution = self.analyze_direction_distribution(motion_vectors)
        total_objects = len(motion_vectors)
        
        # Calculate direction percentages
        direction_percentages = {}
        if total_objects > 0:
            for direction, count in direction_distribution.items():
                direction_percentages[direction] = (count / total_objects) * 100
        
        # Detect bidirectional conflict
        bidirectional_conflict, flow_conflict_score = \
            self.detect_bidirectional_conflict(direction_distribution)
        
        # Detect crowd surge
        surge_detected, surge_direction = \
            self.detect_crowd_surge(direction_distribution, total_objects)
        
        # Calculate speed variation
        speed_variation_score = self.calculate_speed_variation()
        
        # Find dominant direction
        dominant_direction = max(direction_distribution.items(), 
                                key=lambda x: x[1], 
                                default=('Static', 0))[0] if direction_distribution else 'Static'
        
        # Update previous frame data
        self.prev_direction_distribution = direction_distribution.copy()
        self.prev_total_objects = total_objects
        self.bidirectional_conflict_active = bidirectional_conflict
        self.surge_active = surge_detected
        self.surge_direction = surge_direction
        
        return FlowMetrics(
            total_objects=total_objects,
            direction_distribution=direction_distribution,
            direction_percentages=direction_percentages,
            bidirectional_conflict=bidirectional_conflict,
            surge_detected=surge_detected,
            dominant_direction=dominant_direction,
            flow_conflict_score=flow_conflict_score,
            speed_variation_score=speed_variation_score
        )
    
    def get_flow_alerts(self, flow_metrics: FlowMetrics) -> List[str]:
        """
        Generate alerts based on flow analysis.
        
        Args:
            flow_metrics: Flow analysis results
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        if flow_metrics.bidirectional_conflict:
            alerts.append("Bidirectional Flow Conflict Detected")
        
        if flow_metrics.surge_detected:
            direction = flow_metrics.dominant_direction
            alerts.append(f"Sudden Crowd Surge Detected: {direction}")
        
        return alerts
    
    def draw_flow_arrows(self, frame: np.ndarray, tracking_results: List[Dict]) -> np.ndarray:
        """
        Draw motion direction arrows on frame.
        
        Args:
            frame: Input frame
            tracking_results: Tracking results with positions
            
        Returns:
            Frame with flow arrows overlay
        """
        annotated_frame = frame.copy()
        
        for result in tracking_results:
            track_id = result.get('track_id')
            bbox = result.get('bbox')
            
            if track_id is not None and bbox:
                current_x = (bbox[0] + bbox[2]) / 2
                current_y = (bbox[1] + bbox[3]) / 2
                
                vector = self.compute_motion_vector(track_id, current_x, current_y)
                if vector and vector.magnitude >= self.min_movement_threshold:
                    # Draw arrow
                    start_point = (int(current_x), int(current_y))
                    
                    # Scale arrow length based on magnitude
                    arrow_length = min(vector.magnitude * 2, 50)
                    end_x = current_x + (vector.dx / vector.magnitude) * arrow_length if vector.magnitude > 0 else current_x
                    end_y = current_y + (vector.dy / vector.magnitude) * arrow_length if vector.magnitude > 0 else current_y
                    end_point = (int(end_x), int(end_y))
                    
                    # Color based on direction
                    direction_colors = {
                        'Left': (255, 0, 0),
                        'Right': (0, 255, 0),
                        'Up': (0, 0, 255),
                        'Down': (255, 255, 0),
                        'Up-Left': (255, 0, 255),
                        'Up-Right': (0, 255, 255),
                        'Down-Left': (128, 0, 128),
                        'Down-Right': (0, 128, 128)
                    }
                    color = direction_colors.get(vector.direction, (255, 255, 255))
                    
                    # Draw arrow
                    cv2.arrowedLine(annotated_frame, start_point, end_point, color, 2)
        
        return annotated_frame
    
    def get_flow_summary(self, flow_metrics: FlowMetrics) -> Dict:
        """
        Get comprehensive flow analysis summary.
        
        Args:
            flow_metrics: Flow analysis results
            
        Returns:
            Dictionary with flow summary
        """
        return {
            'total_tracked_objects': flow_metrics.total_objects,
            'dominant_direction': flow_metrics.dominant_direction,
            'direction_percentages': flow_metrics.direction_percentages,
            'bidirectional_conflict': flow_metrics.bidirectional_conflict,
            'flow_conflict_score': flow_metrics.flow_conflict_score,
            'surge_detected': flow_metrics.surge_detected,
            'surge_direction': self.surge_direction if flow_metrics.surge_detected else None,
            'speed_variation_score': flow_metrics.speed_variation_score,
            'active_alerts': self.get_flow_alerts(flow_metrics)
        }
