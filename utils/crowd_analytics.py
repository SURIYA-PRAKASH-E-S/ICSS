"""
Crowd Analytics Module
Enhanced crowd level classification, metrics tracking, and visualization
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional
from collections import deque
import time


class CrowdAnalytics:
    """
    Crowd analytics for level classification, peak tracking, and density analysis
    """
    
    def __init__(
        self,
        low_count_threshold: int = 5,
        medium_count_threshold: int = 15,
        low_density_threshold: float = 0.3,
        medium_density_threshold: float = 0.7,
        history_size: int = 100
    ):
        self.low_count_threshold = low_count_threshold
        self.medium_count_threshold = medium_count_threshold
        self.low_density_threshold = low_density_threshold
        self.medium_density_threshold = medium_density_threshold
        
        # Historical data for peak/average tracking
        self.count_history = deque(maxlen=history_size)
        self.density_history = deque(maxlen=history_size)
        
        # Peak tracking
        self.peak_count = 0
        self.peak_density = 0.0
        self.peak_timestamp = None
        self.total_frames = 0
        self.running_sum_count = 0
        self.running_sum_density = 0.0
        
        # Color mappings (BGR for OpenCV)
        self.crowd_colors = {
            'Low': (0, 255, 0),      # Green
            'Medium': (0, 255, 255), # Yellow
            'High': (0, 0, 255)      # Red
        }
        
        # Heatmap accumulator
        self.heatmap = None
        self.heatmap_alpha = 0.5
    
    def classify_crowd_level(
        self,
        people_count: int,
        density: float,
        use_count: bool = True,
        use_density: bool = True
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        Classify crowd level based on count and/or density
        
        Args:
            people_count: Number of people detected
            density: Crowd density (0-1 scale)
            use_count: Use count for classification
            use_density: Use density for classification
            
        Returns:
            Tuple of (crowd_level, color_bgr)
        """
        count_level = 'Low'
        density_level = 'Low'
        
        if use_count:
            if people_count >= self.medium_count_threshold:
                count_level = 'High'
            elif people_count >= self.low_count_threshold:
                count_level = 'Medium'
        
        if use_density:
            if density >= self.medium_density_threshold:
                density_level = 'High'
            elif density >= self.low_density_threshold:
                density_level = 'Medium'
        
        # Use the higher of the two levels
        level_priority = {'Low': 0, 'Medium': 1, 'High': 2}
        levels = [count_level if use_count else 'Low', density_level if use_density else 'Low']
        max_level = max(levels, key=lambda x: level_priority[x])
        
        return max_level, self.crowd_colors[max_level]
    
    def update_metrics(
        self,
        people_count: int,
        density: float,
        tracked_objects: list = None
    ) -> Dict[str, Any]:
        """
        Update analytics with new frame data
        
        Args:
            people_count: Current people count
            density: Current density
            tracked_objects: List of tracked object centroids
            
        Returns:
            Dictionary with current metrics
        """
        self.total_frames += 1
        self.running_sum_count += people_count
        self.running_sum_density += density
        
        # Update history
        self.count_history.append(people_count)
        self.density_history.append(density)
        
        # Update peak values
        if people_count > self.peak_count:
            self.peak_count = people_count
            self.peak_timestamp = time.time()
        
        if density > self.peak_density:
            self.peak_density = density
        
        # Get crowd level
        crowd_level, crowd_color = self.classify_crowd_level(people_count, density)
        
        # Calculate average
        avg_count = self.running_sum_count / self.total_frames if self.total_frames > 0 else 0
        avg_density = self.running_sum_density / self.total_frames if self.total_frames > 0 else 0
        
        metrics = {
            'people_count': people_count,
            'density': density,
            'crowd_level': crowd_level,
            'crowd_color': crowd_color,
            'peak_count': self.peak_count,
            'peak_density': self.peak_density,
            'average_count': avg_count,
            'average_density': avg_density,
            'total_frames': self.total_frames,
            'peak_timestamp': self.peak_timestamp
        }
        
        return metrics
    
    def update_heatmap(
        self,
        frame_shape: Tuple[int, int],
        tracked_objects: list,
        bbox_list: list = None
    ) -> np.ndarray:
        """
        Update and return crowd density heatmap
        
        Args:
            frame_shape: (height, width) of the frame
            tracked_objects: List of tracked object centroids (cx, cy)
            bbox_list: Optional list of bounding boxes for density calculation
            
        Returns:
            Heatmap array normalized to 0-1
        """
        h, w = frame_shape[:2]
        
        # Initialize heatmap if needed
        if self.heatmap is None or self.heatmap.shape != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Add Gaussian influence for each tracked person
        if tracked_objects:
            for obj in tracked_objects:
                if isinstance(obj, dict):
                    cx, cy = obj.get('cx', 0), obj.get('cy', 0)
                else:
                    cx, cy = obj[0], obj[1]
                
                # Add Gaussian blob at person location
                if 0 <= cx < w and 0 <= cy < h:
                    y, x = np.ogrid[:h, :w]
                    radius = 30
                    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
                    self.heatmap[mask] += np.exp(
                        -((x[mask] - cx)**2 + (y[mask] - cy)**2) / (2 * (radius/2)**2)
                    )
        
        # Decay heatmap slightly for temporal smoothing
        self.heatmap = self.heatmap * 0.95
        
        # Normalize to 0-1
        if self.heatmap.max() > 0:
            self.heatmap = self.heatmap / self.heatmap.max()
        
        return self.heatmap.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.count_history.clear()
        self.density_history.clear()
        self.peak_count = 0
        self.peak_density = 0.0
        self.peak_timestamp = None
        self.total_frames = 0
        self.running_sum_count = 0
        self.running_sum_density = 0.0
        self.heatmap = None


def draw_crowd_overlay(
    frame: np.ndarray,
    metrics: Dict[str, Any],
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw clean crowd information overlay on frame
    
    Args:
        frame: Input frame
        metrics: Metrics from CrowdAnalytics.update_metrics()
        position: Starting position (x, y)
        font_scale: Font scale for text
        
    Returns:
        Frame with overlay
    """
    annotated = frame.copy()
    x, y = position
    
    # Background panel
    panel_width = 220
    panel_height = 130
    cv2.rectangle(annotated, (x-5, y-25), (x+panel_width, y+panel_height), (0, 0, 0), -1)
    cv2.rectangle(annotated, (x-5, y-25), (x+panel_width, y+panel_height), (200, 200, 200), 1)
    
    # Get crowd color
    crowd_color = metrics.get('crowd_color', (0, 255, 0))
    
    # Draw crowd level indicator (colored circle)
    cv2.circle(annotated, (x+15, y), 10, crowd_color, -1)
    cv2.circle(annotated, (x+15, y), 12, (255, 255, 255), 1)
    
    # Crowd level text
    level_text = f"Level: {metrics['crowd_level']}"
    cv2.putText(annotated, level_text, (x+30, y+5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Count
    count_text = f"Count: {metrics['people_count']}"
    cv2.putText(annotated, count_text, (x, y+30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Density
    density_text = f"Density: {metrics['density']:.3f}"
    cv2.putText(annotated, density_text, (x, y+55),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    # Peak
    peak_text = f"Peak: {metrics['peak_count']}"
    cv2.putText(annotated, peak_text, (x, y+80),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 2)
    
    # Average
    avg_text = f"Avg: {metrics['average_count']:.1f}"
    cv2.putText(annotated, avg_text, (x, y+105),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 2)
    
    return annotated


def draw_heatmap_overlay(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on frame with proper blending
    
    Args:
        frame: Input frame
        heatmap: Normalized heatmap (0-1)
        alpha: Blend transparency
        colormap: OpenCV colormap
        
    Returns:
        Frame with heatmap overlay
    """
    # Resize heatmap to match frame if needed
    if heatmap.shape[:2] != frame.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    # Convert to uint8 for colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Blend with frame
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlay


def draw_legend(
    frame: np.ndarray,
    position: Tuple[int, int] = (10, 100),
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw crowd level legend on frame
    
    Args:
        frame: Input frame
        position: Starting position
        font_scale: Font scale
        
    Returns:
        Frame with legend
    """
    annotated = frame.copy()
    x, y = position
    
    legend_items = [
        ("Low", (0, 255, 0)),
        ("Medium", (0, 255, 255)),
        ("High", (0, 0, 255))
    ]
    
    # Background
    cv2.rectangle(annotated, (x-5, y-15), (x+120, y+len(legend_items)*25+5), (0, 0, 0), -1)
    
    for i, (label, color) in enumerate(legend_items):
        cy = y + i * 25
        cv2.rectangle(annotated, (x, cy-10), (x+20, cy+10), color, -1)
        cv2.putText(annotated, label, (x+25, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return annotated


def create_zone_heatmap(
    frame_shape: Tuple[int, int],
    tracked_objects: list,
    grid_size: Tuple[int, int] = (3, 3)
) -> Dict[str, Any]:
    """
    Create zone-based density visualization
    
    Args:
        frame_shape: (height, width)
        tracked_objects: List of tracked centroids
        grid_size: (rows, cols)
        
    Returns:
        Dictionary with zone densities and colors
    """
    h, w = frame_shape[:2]
    rows, cols = grid_size
    
    zone_h = h // rows
    zone_w = w // cols
    
    zones = {}
    
    # Count people in each zone
    zone_counts = np.zeros((rows, cols), dtype=np.int32)
    
    for obj in tracked_objects:
        if isinstance(obj, dict):
            cx, cy = obj.get('cx', 0), obj.get('cy', 0)
        else:
            cx, cy = obj[0], obj[1]
        
        if 0 <= cx < w and 0 <= cy < h:
            col_idx = min(cx // zone_w, cols - 1)
            row_idx = min(cy // zone_h, rows - 1)
            zone_counts[row_idx, col_idx] += 1
    
    # Create zone data
    max_count = zone_counts.max() if zone_counts.max() > 0 else 1
    
    for row in range(rows):
        for col in range(cols):
            zone_id = f"{row}_{col}"
            count = zone_counts[row, col]
            density = count / max_count if max_count > 0 else 0
            
            # Color based on density
            if density < 0.33:
                color = (0, 255, 0)   # Green
            elif density < 0.66:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255)   # Red
            
            zones[zone_id] = {
                'bbox': (col * zone_w, row * zone_h, (col+1) * zone_w, (row+1) * zone_h),
                'count': count,
                'density': density,
                'color': color
            }
    
    return zones


def draw_zone_grid(
    frame: np.ndarray,
    zones: Dict[str, Any],
    line_thickness: int = 2,
    show_labels: bool = True,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw zone grid with density colors
    
    Args:
        frame: Input frame
        zones: Zone data from create_zone_heatmap
        line_thickness: Border thickness
        show_labels: Show count labels
        font_scale: Font scale
        
    Returns:
        Frame with zone overlay
    """
    annotated = frame.copy()
    
    for zone_id, zone_data in zones.items():
        x1, y1, x2, y2 = zone_data['bbox']
        color = zone_data['color']
        count = zone_data['count']
        
        # Draw zone border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
        
        # Semi-transparent fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        
        if show_labels and count > 0:
            label = str(count)
            label_pos = ((x1+x2)//2 - 10, (y1+y2)//2 + 5)
            cv2.putText(annotated, label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    
    return annotated


def init_crowd_analytics(**kwargs) -> CrowdAnalytics:
    """Initialize CrowdAnalytics instance"""
    return CrowdAnalytics(**kwargs)
