"""
Smart Risk Engine for Intelligent Crowd Analytics
Dynamic weighted risk model with explainable AI alerts
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple, List, Optional

class SmartRiskEngine:
    """
    Advanced risk assessment engine with dynamic weighted scoring
    and explainable AI alerts for crowd surveillance.
    """
    
    def __init__(self, 
                 w_density: float = 0.4, 
                 w_flow_conflict: float = 0.35, 
                 w_speed_variation: float = 0.25,
                 smoothing_window: int = 5,
                 density_threshold_low: float = 0.3,
                 density_threshold_high: float = 0.8,
                 flow_conflict_threshold: float = 0.4,
                 speed_variation_threshold: float = 0.5):
        """
        Initialize Smart Risk Engine with configurable weights and thresholds.
        
        Args:
            w_density: Weight for crowd density factor
            w_flow_conflict: Weight for flow conflict factor  
            w_speed_variation: Weight for speed variation factor
            smoothing_window: Moving average window for risk smoothing
            density_threshold_low: Low density threshold for risk calculation
            density_threshold_high: High density threshold for risk calculation
            flow_conflict_threshold: Threshold for flow conflict detection
            speed_variation_threshold: Threshold for speed variation detection
        """
        self.w_density = w_density
        self.w_flow_conflict = w_flow_conflict
        self.w_speed_variation = w_speed_variation
        
        # Thresholds
        self.density_threshold_low = density_threshold_low
        self.density_threshold_high = density_threshold_high
        self.flow_conflict_threshold = flow_conflict_threshold
        self.speed_variation_threshold = speed_variation_threshold
        
        # Smoothing
        self.smoothing_window = smoothing_window
        self.risk_history = deque(maxlen=smoothing_window)
        
        # Risk level definitions
        self.risk_levels = {
            'Normal': {'range': (0.0, 0.4), 'color': (0, 255, 0), 'priority': 1},
            'Moderate': {'range': (0.4, 0.7), 'color': (0, 255, 255), 'priority': 2},
            'High Risk': {'range': (0.7, 1.0), 'color': (0, 0, 255), 'priority': 3}
        }
    
    def normalize_inputs(self, 
                       density: float, 
                       flow_conflict: float, 
                       speed_variation: float) -> Tuple[float, float, float]:
        """
        Normalize all risk inputs to [0, 1] range.
        
        Args:
            density: Crowd density (people per unit area)
            flow_conflict: Flow conflict score
            speed_variation: Speed variation score
            
        Returns:
            Normalized tuple (density_norm, flow_conflict_norm, speed_variation_norm)
        """
        # Normalize density based on thresholds
        if density <= self.density_threshold_low:
            density_norm = 0.0
        elif density >= self.density_threshold_high:
            density_norm = 1.0
        else:
            # Linear interpolation between thresholds
            density_norm = (density - self.density_threshold_low) / \
                          (self.density_threshold_high - self.density_threshold_low)
        
        # Flow conflict is already normalized [0, 1]
        flow_conflict_norm = np.clip(flow_conflict, 0.0, 1.0)
        
        # Speed variation is already normalized [0, 1]
        speed_variation_norm = np.clip(speed_variation, 0.0, 1.0)
        
        return density_norm, flow_conflict_norm, speed_variation_norm
    
    def compute_risk_score(self, 
                          density: float, 
                          flow_conflict: float, 
                          speed_variation: float) -> float:
        """
        Compute weighted risk score from normalized inputs.
        
        Args:
            density: Crowd density
            flow_conflict: Flow conflict score
            speed_variation: Speed variation score
            
        Returns:
            Weighted risk score [0, 1]
        """
        # Normalize inputs
        density_norm, flow_conflict_norm, speed_variation_norm = \
            self.normalize_inputs(density, flow_conflict, speed_variation)
        
        # Compute weighted risk score
        risk_score = (self.w_density * density_norm + 
                     self.w_flow_conflict * flow_conflict_norm + 
                     self.w_speed_variation * speed_variation_norm)
        
        # Ensure score is in [0, 1] range
        risk_score = np.clip(risk_score, 0.0, 1.0)
        
        return risk_score
    
    def smooth_risk_score(self, current_score: float) -> float:
        """
        Apply moving average smoothing to risk score to avoid flickering.
        
        Args:
            current_score: Current risk score
            
        Returns:
            Smoothed risk score
        """
        self.risk_history.append(current_score)
        
        if len(self.risk_history) < self.smoothing_window:
            # Not enough history, return current score
            return current_score
        
        # Compute moving average
        smoothed_score = np.mean(self.risk_history)
        return smoothed_score
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level from risk score.
        
        Args:
            risk_score: Risk score [0, 1]
            
        Returns:
            Risk level string
        """
        for level, config in self.risk_levels.items():
            min_score, max_score = config['range']
            if min_score <= risk_score < max_score:
                return level
        
        # Default to highest risk if score is at maximum
        return 'High Risk'
    
    def get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        """
        Get color associated with risk level.
        
        Args:
            risk_level: Risk level string
            
        Returns:
            RGB color tuple
        """
        return self.risk_levels.get(risk_level, {'color': (128, 128, 128)})['color']
    
    def generate_explainable_alert(self, 
                                 density: float, 
                                 flow_conflict: float, 
                                 speed_variation: float,
                                 risk_score: float,
                                 risk_level: str) -> str:
        """
        Generate explainable AI alert based on risk factors.
        
        Args:
            density: Crowd density
            flow_conflict: Flow conflict score
            speed_variation: Speed variation score
            risk_score: Computed risk score
            risk_level: Determined risk level
            
        Returns:
            Explainable alert message
        """
        # Normalize inputs for display
        density_norm, flow_conflict_norm, speed_variation_norm = \
            self.normalize_inputs(density, flow_conflict, speed_variation)
        
        # Identify primary contributing factors
        factors = []
        
        if density_norm > 0.6:
            factors.append(f"high density ({density_norm:.2f})")
        elif density_norm > 0.3:
            factors.append(f"medium density ({density_norm:.2f})")
        
        if flow_conflict_norm > 0.5:
            factors.append(f"strong opposite flow ({flow_conflict_norm:.2f})")
        elif flow_conflict_norm > 0.3:
            factors.append(f"moderate flow conflict ({flow_conflict_norm:.2f})")
        
        if speed_variation_norm > 0.5:
            factors.append(f"high speed variation ({speed_variation_norm:.2f})")
        elif speed_variation_norm > 0.3:
            factors.append(f"moderate speed variation ({speed_variation_norm:.2f})")
        
        # Generate alert based on risk level and factors
        if risk_level == 'High Risk':
            if factors:
                return f"High Risk: {' + '.join(factors[:2])} detected"
            else:
                return "High Risk: Multiple risk factors detected"
        elif risk_level == 'Moderate':
            if factors:
                return f"Moderate Risk: {' + '.join(factors[:2])} detected"
            else:
                return "Moderate Risk: Some risk factors present"
        else:
            return "Normal: No significant risk factors detected"
    
    def compute_comprehensive_risk(self, 
                                 density: float, 
                                 flow_conflict: float, 
                                 speed_variation: float) -> Dict:
        """
        Compute comprehensive risk assessment with all outputs.
        
        Args:
            density: Crowd density
            flow_conflict: Flow conflict score
            speed_variation: Speed variation score
            
        Returns:
            Dictionary with risk assessment results
        """
        # Compute raw risk score
        raw_risk_score = self.compute_risk_score(density, flow_conflict, speed_variation)
        
        # Apply smoothing
        smoothed_risk_score = self.smooth_risk_score(raw_risk_score)
        
        # Determine risk level
        risk_level = self.get_risk_level(smoothed_risk_score)
        
        # Get risk color
        risk_color = self.get_risk_color(risk_level)
        
        # Generate explainable alert
        alert = self.generate_explainable_alert(
            density, flow_conflict, speed_variation, 
            smoothed_risk_score, risk_level
        )
        
        # Normalize inputs for display
        density_norm, flow_conflict_norm, speed_variation_norm = \
            self.normalize_inputs(density, flow_conflict, speed_variation)
        
        return {
            'risk_score': smoothed_risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_alert': alert,
            'normalized_inputs': {
                'density': density_norm,
                'flow_conflict': flow_conflict_norm,
                'speed_variation': speed_variation_norm
            },
            'raw_factors': {
                'density': density,
                'flow_conflict': flow_conflict,
                'speed_variation': speed_variation
            }
        }
    
    def update_weights(self, 
                     w_density: float = None, 
                     w_flow_conflict: float = None, 
                     w_speed_variation: float = None):
        """
        Update risk weights dynamically.
        
        Args:
            w_density: New density weight
            w_flow_conflict: New flow conflict weight
            w_speed_variation: New speed variation weight
        """
        if w_density is not None:
            self.w_density = np.clip(w_density, 0.0, 1.0)
        if w_flow_conflict is not None:
            self.w_flow_conflict = np.clip(w_flow_conflict, 0.0, 1.0)
        if w_speed_variation is not None:
            self.w_speed_variation = np.clip(w_speed_variation, 0.0, 1.0)
        
        # Normalize weights to sum to 1
        total_weight = self.w_density + self.w_flow_conflict + self.w_speed_variation
        if total_weight > 0:
            self.w_density /= total_weight
            self.w_flow_conflict /= total_weight
            self.w_speed_variation /= total_weight
    
    def reset_history(self):
        """Reset risk history for new session."""
        self.risk_history.clear()
