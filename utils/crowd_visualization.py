"""
Crowd Visualization Module
Handles heatmap overlay, zone highlighting, and risk visualization
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class CrowdVisualizer:
    """
    Visualization handler for dense crowd detection results
    """
    
    def __init__(self):
        """Initialize visualizer with custom colormaps"""
        self.custom_colormap = self._create_custom_colormap()
    
    def _create_custom_colormap(self):
        """Create custom colormap for density visualization"""
        colors = [
            (0, 0, 0.5),      # Dark blue (low density)
            (0, 0, 1),        # Blue
            (0, 1, 1),        # Cyan
            (0, 1, 0),        # Green
            (1, 1, 0),        # Yellow
            (1, 0.5, 0),      # Orange
            (1, 0, 0)         # Red (high density)
        ]
        return LinearSegmentedColormap.from_list('crowd_density', colors, N=256)
    
    def draw_zone_grid(
        self,
        frame: np.ndarray,
        zones: Dict[str, Any],
        line_thickness: int = 2,
        show_labels: bool = True,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw zone grid with color-coded risk levels
        
        Args:
            frame: Input frame
            zones: Zone dictionary with density and risk info
            line_thickness: Thickness of zone borders
            show_labels: Show zone density labels
            font_scale: Font scale for labels
            
        Returns:
            Frame with zone grid overlay
        """
        annotated_frame = frame.copy()
        
        for zone_id, zone_data in zones.items():
            x1, y1, x2, y2 = zone_data['bbox']
            color = zone_data['color']
            density = zone_data['density']
            risk_level = zone_data['risk_level']
            
            # Draw zone boundary
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Fill zone with semi-transparent color
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
            
            if show_labels:
                # Draw zone label
                label = f"Z{zone_id.split('_')[1]}: {density:.2f}"
                label_pos = (x1 + 5, y1 + 20)
                
                # Background for label
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (label_pos[0] - 2, label_pos[1] - text_height - 2),
                    (label_pos[0] + text_width + 2, label_pos[1] + baseline + 2),
                    (0, 0, 0), -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
                )
        
        return annotated_frame
    
    def draw_bounding_boxes(
        self,
        frame: np.ndarray,
        boxes: List[Tuple],
        confidences: List[float] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            boxes: List of bounding boxes (x1, y1, x2, y2)
            confidences: Confidence scores for each box
            color: Box color (BGR)
            thickness: Line thickness
            show_confidence: Show confidence scores
            
        Returns:
            Frame with bounding boxes
        """
        annotated_frame = frame.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            if show_confidence and confidences and i < len(confidences):
                # Draw confidence score
                label = f"{confidences[i]:.2f}"
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
        
        return annotated_frame
    
    def overlay_density_heatmap(
        self,
        frame: np.ndarray,
        density_map: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay density heatmap on frame
        
        Args:
            frame: Input frame
            density_map: Normalized density map (0-1)
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Frame with heatmap overlay
        """
        # Normalize density map to 0-255
        density_normalized = (density_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(density_normalized, colormap)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def draw_info_panel(
        self,
        frame: np.ndarray,
        info_dict: Dict[str, Any],
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            info_dict: Dictionary of information to display
            position: Starting position (x, y)
            font_scale: Font scale
            color: Text color
            thickness: Text thickness
            bg_color: Background color
            
        Returns:
            Frame with info panel
        """
        annotated_frame = frame.copy()
        x, y = position
        
        # Calculate panel size
        max_width = 0
        total_height = 0
        line_height = 25
        
        lines = []
        for key, value in info_dict.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"
            
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            max_width = max(max_width, text_width)
            total_height += line_height
            lines.append(text)
        
        # Draw background panel
        cv2.rectangle(
            annotated_frame,
            (x - 5, y - 20),
            (x + max_width + 10, y + total_height),
            bg_color, -1
        )
        
        # Draw text lines
        current_y = y
        for line in lines:
            cv2.putText(
                annotated_frame, line, (x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
            )
            current_y += line_height
        
        return annotated_frame
    
    def draw_risk_legend(
        self,
        frame: np.ndarray,
        position: Tuple[int, int] = (10, 100),
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw risk level legend on frame
        
        Args:
            frame: Input frame
            position: Starting position (x, y)
            font_scale: Font scale
            
        Returns:
            Frame with legend
        """
        annotated_frame = frame.copy()
        x, y = position
        
        # Legend items
        legend_items = [
            ("Low Density", (0, 255, 0)),      # Green
            ("Medium Density", (0, 255, 255)), # Yellow
            ("High Density", (0, 0, 255))      # Red
        ]
        
        # Draw legend background
        cv2.rectangle(
            annotated_frame,
            (x - 5, y - 15),
            (x + 150, y + len(legend_items) * 25 + 5),
            (0, 0, 0), -1
        )
        
        # Draw legend items
        current_y = y
        for label, color in legend_items:
            # Draw color box
            cv2.rectangle(
                annotated_frame,
                (x, current_y - 10),
                (x + 20, current_y + 10),
                color, -1
            )
            
            # Draw label
            cv2.putText(
                annotated_frame, label, (x + 25, current_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
            )
        
        # Draw risk legend
        annotated_frame = self.draw_risk_legend(
            annotated_frame, position=(10, frame.shape[0] - 100)
        )
        
        return annotated_frame
    
    def generate_density_histogram(
        self,
        density_map: np.ndarray,
        figsize: Tuple[int, int] = (8, 4)
    ) -> np.ndarray:
        """
        Generate density histogram plot
        
        Args:
            density_map: Density map
            figsize: Figure size (width, height)
            
        Returns:
            Histogram image as numpy array
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Flatten density map for histogram
        density_flat = density_map.flatten()
        
        # Plot histogram
        ax.hist(density_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Density Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Density Distribution')
        ax.grid(True, alpha=0.3)
        
        # Convert plot to numpy array
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return plot_img


def init_visualizer() -> CrowdVisualizer:
    """Initialize crowd visualizer"""
    return CrowdVisualizer()
