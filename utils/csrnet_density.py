"""
CSRNet Density Estimation Module
Generates density heatmaps and estimates crowd count from video frames
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy import stats

class CSRNet(nn.Module):
    """
    CSRNet (Congested Scene Recognition Network) for crowd density estimation
    Uses VGG-16 frontend and dilated convolution backend
    """
    
    def __init__(self, pretrained: bool = True):
        super(CSRNet, self).__init__()
        
        # Frontend (VGG-16 first 10 layers)
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Backend (Dilated convolutions)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """
        Load pretrained weights if available
        For now, use random initialization
        In production, download from official CSRNet repository
        """
        pass


class DensityEstimator:
    """
    Density estimation module using CSRNet or fallback methods
    """
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Initialize density estimator
        
        Args:
            model_path: Path to pretrained CSRNet weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.use_csrnet = False
        
        try:
            # Initialize CSRNet model
            self.model = CSRNet(pretrained=True)
            self.model.to(device)
            self.model.eval()
            
            # Load custom weights if provided
            if model_path:
                checkpoint = torch.load(model_path, map_location=device)
                self.model.load_state_dict(checkpoint)
            
            self.use_csrnet = True
            print("CSRNet density estimator initialized")
            
        except Exception as e:
            print(f"CSRNet initialization failed: {e}")
            print("Using fallback density estimation method")
            self.use_csrnet = False
    
    def estimate_density(
        self,
        frame: np.ndarray,
        detections: list = None,
        use_gaussian: bool = True
    ) -> Dict[str, Any]:
        """
        Estimate crowd density from frame
        
        Args:
            frame: Input frame
            detections: List of bounding boxes (x1, y1, x2, y2)
            use_gaussian: Use Gaussian smoothing for density map
            
        Returns:
            Dictionary containing density map and estimated count
        """
        if self.use_csrnet:
            return self._estimate_with_csrnet(frame, use_gaussian)
        else:
            return self._estimate_with_detections(frame, detections, use_gaussian)
    
    def _estimate_with_csrnet(self, frame: np.ndarray, use_gaussian: bool) -> Dict[str, Any]:
        """
        Estimate density using CSRNet model
        
        Args:
            frame: Input frame
            use_gaussian: Apply Gaussian smoothing
            
        Returns:
            Dictionary with density map and count
        """
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                density_map = self.model(input_tensor)
            
            # Convert to numpy
            density_map = density_map.squeeze().cpu().numpy()
            
            # Resize to original frame size
            density_map = cv2.resize(
                density_map,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Apply Gaussian smoothing
            if use_gaussian:
                density_map = gaussian_filter(density_map, sigma=1)
            
            # Normalize density map
            if density_map.max() > 0:
                density_map_normalized = density_map / density_map.max()
            else:
                density_map_normalized = density_map
            
            # Estimate total count
            estimated_count = density_map.sum()
            
            return {
                'density_map': density_map,
                'density_map_normalized': density_map_normalized,
                'estimated_count': estimated_count,
                'method': 'csrnet'
            }
            
        except Exception as e:
            print(f"CSRNet inference error: {e}")
            return self._estimate_with_detections(frame, None, use_gaussian)
    
    def _estimate_with_detections(
        self,
        frame: np.ndarray,
        detections: list,
        use_gaussian: bool
    ) -> Dict[str, Any]:
        """
        Fallback density estimation using detection centroids
        
        Args:
            frame: Input frame
            detections: List of bounding boxes
            use_gaussian: Apply Gaussian smoothing
            
        Returns:
            Dictionary with density map and count
        """
        h, w = frame.shape[:2]
        
        # Initialize density map
        density_map = np.zeros((h, w), dtype=np.float32)
        
        if detections and len(detections) > 0:
            # Use detection centroids to create density map
            for bbox in detections:
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Add Gaussian blob at each centroid
                if 0 <= cx < w and 0 <= cy < h:
                    # Create small Gaussian kernel
                    sigma = 15
                    x = np.arange(max(0, cx - 3*sigma), min(w, cx + 3*sigma))
                    y = np.arange(max(0, cy - 3*sigma), min(h, cy + 3*sigma))
                    xx, yy = np.meshgrid(x, y)
                    
                    # Gaussian function
                    gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
                    
                    # Add to density map
                    density_map[
                        max(0, cy - 3*sigma):min(h, cy + 3*sigma),
                        max(0, cx - 3*sigma):min(w, cx + 3*sigma)
                    ] += gaussian
        
        # Apply Gaussian smoothing
        if use_gaussian:
            density_map = gaussian_filter(density_map, sigma=5)
        
        # Normalize
        if density_map.max() > 0:
            density_map_normalized = density_map / density_map.max()
        else:
            density_map_normalized = density_map
        
        # Estimate count
        estimated_count = len(detections) if detections else 0
        
        return {
            'density_map': density_map,
            'density_map_normalized': density_map_normalized,
            'estimated_count': estimated_count,
            'method': 'detection_based'
        }
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for CSRNet inference
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Subtract mean (ImageNet values)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor
    
    def generate_heatmap(
        self,
        density_map: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate colored heatmap from density map
        
        Args:
            density_map: Normalized density map
            colormap: OpenCV colormap
            
        Returns:
            Colored heatmap image
        """
        # Normalize to 0-255
        density_normalized = (density_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(density_normalized, colormap)
        
        return heatmap
    
    def overlay_heatmap_on_frame(
        self,
        frame: np.ndarray,
        density_map: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay density heatmap on original frame
        
        Args:
            frame: Original frame
            density_map: Density map
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Frame with heatmap overlay
        """
        # Generate heatmap
        heatmap = self.generate_heatmap(density_map, colormap)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def calculate_zone_density(
        self,
        density_map: np.ndarray,
        grid_size: tuple = (3, 3)
    ) -> Dict[str, Any]:
        """
        Calculate average density per zone in grid
        
        Args:
            density_map: Density map
            grid_size: Grid dimensions (rows, cols)
            
        Returns:
            Dictionary with zone densities and risk levels
        """
        h, w = density_map.shape
        rows, cols = grid_size
        
        zone_height = h // rows
        zone_width = w // cols
        
        zones = {}
        zone_id = 0
        
        for i in range(rows):
            for j in range(cols):
                y1 = i * zone_height
                y2 = (i + 1) * zone_height if i < rows - 1 else h
                x1 = j * zone_width
                x2 = (j + 1) * zone_width if j < cols - 1 else w
                
                # Calculate average density in zone
                zone_density = density_map[y1:y2, x1:x2].mean()
                
                # Classify zone risk
                if zone_density < 0.3:
                    risk_level = "Low"
                    color = (0, 255, 0)  # Green
                elif zone_density < 0.6:
                    risk_level = "Medium"
                    color = (0, 255, 255)  # Yellow
                else:
                    risk_level = "High"
                    color = (0, 0, 255)  # Red
                
                zones[f"zone_{zone_id}"] = {
                    'density': zone_density,
                    'risk_level': risk_level,
                    'color': color,
                    'bbox': (x1, y1, x2, y2),
                    'grid_position': (i, j)
                }
                
                zone_id += 1
        
        return zones
    
    def is_available(self) -> bool:
        """Check if CSRNet is available"""
        return self.use_csrnet


def init_density_estimator(model_path: str = None, device: str = "cpu") -> DensityEstimator:
    """
    Initialize density estimator
    
    Args:
        model_path: Path to pretrained weights
        device: Device to run inference on
        
    Returns:
        DensityEstimator instance
    """
    return DensityEstimator(model_path=model_path, device=device)
