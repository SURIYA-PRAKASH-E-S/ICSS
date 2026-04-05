"""
Advanced Analytics Integration for Intelligent Crowd Surveillance
Integrates Risk Engine, Zone Analyzer, and Flow Analyzer
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from .risk_engine import SmartRiskEngine
from .zone_analyzer import ZoneAnalyzer
from .flow_analyzer import FlowAnalyzer

class AdvancedCrowdAnalytics:
    """
    Advanced crowd analytics platform that integrates risk assessment,
    zone monitoring, and flow analysis for intelligent surveillance.
    """
    
    def __init__(self, 
                 frame_width: int = 640,
                 frame_height: int = 480,
                 grid_size: Tuple[int, int] = (2, 2),
                 risk_weights: Dict[str, float] = None,
                 restricted_zones: List[str] = None):
        """
        Initialize Advanced Crowd Analytics platform.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            grid_size: Zone grid size (rows, cols)
            risk_weights: Risk engine weights
            restricted_zones: List of restricted zone IDs
        """
        # Initialize components
        self.risk_engine = SmartRiskEngine(
            w_density=risk_weights.get('density', 0.4) if risk_weights else 0.4,
            w_flow_conflict=risk_weights.get('flow_conflict', 0.35) if risk_weights else 0.35,
            w_speed_variation=risk_weights.get('speed_variation', 0.25) if risk_weights else 0.25
        )
        
        self.zone_analyzer = ZoneAnalyzer(
            frame_width=frame_width,
            frame_height=frame_height,
            grid_rows=grid_size[0],
            grid_cols=grid_size[1],
            real_world_width_m=50.0,
            real_world_height_m=30.0
        )
        
        self.flow_analyzer = FlowAnalyzer()
        
        # Frame dimensions
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Analytics state
        self.analytics_enabled = True
        self.zone_analysis_enabled = True
        self.flow_analysis_enabled = True
        self.risk_analysis_enabled = True
    
    def extract_detections_from_tracking(self, tracking_results: List[Dict]) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes from tracking results.
        
        Args:
            tracking_results: List of tracking results
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        detections = []
        for result in tracking_results:
            bbox = result.get('bbox')
            if bbox:
                detections.append(tuple(bbox))
        return detections
    
    def compute_basic_metrics(self, tracking_results: List[Dict]) -> Dict:
        """
        Compute basic crowd metrics.
        
        Args:
            tracking_results: List of tracking results
            
        Returns:
            Dictionary with basic metrics
        """
        people_count = len(tracking_results)
        frame_area = self.frame_width * self.frame_height
        density = people_count / frame_area if frame_area > 0 else 0
        
        return {
            'people_count': people_count,
            'density': density,
            'frame_area': frame_area
        }
    
    def process_frame_advanced(self, 
                              frame: np.ndarray, 
                              tracking_results: List[Dict],
                              show_zones: bool = True,
                              show_flow_arrows: bool = True,
                              show_risk_banner: bool = True) -> Dict:
        """
        Process frame with advanced analytics pipeline.
        
        Args:
            frame: Input frame
            tracking_results: List of tracking results
            show_zones: Whether to show zone overlay
            show_flow_arrows: Whether to show flow arrows
            show_risk_banner: Whether to show risk banner
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if not self.analytics_enabled:
            return self._basic_processing(frame, tracking_results)
        
        # Extract basic metrics
        basic_metrics = self.compute_basic_metrics(tracking_results)
        
        # Extract detections for zone analysis
        detections = self.extract_detections_from_tracking(tracking_results)
        
        # Initialize results
        analysis_results = {
            'basic_metrics': basic_metrics,
            'risk_analysis': None,
            'zone_analysis': None,
            'flow_analysis': None,
            'alerts': [],
            'annotated_frame': frame.copy()
        }
        
        # Zone Analysis
        if self.zone_analysis_enabled:
            zone_stats = self.zone_analyzer.update_zones_with_detections(detections)
            zone_summary = self.zone_analyzer.get_zone_summary()
            zone_alerts = self.zone_analyzer.get_zone_alerts()
            
            analysis_results['zone_analysis'] = {
                'stats': zone_stats,
                'summary': zone_summary,
                'alerts': zone_alerts
            }
            analysis_results['alerts'].extend(zone_alerts)
        
        # Flow Analysis
        if self.flow_analysis_enabled:
            flow_metrics = self.flow_analyzer.analyze_flow(tracking_results)
            flow_summary = self.flow_analyzer.get_flow_summary(flow_metrics)
            flow_alerts = self.flow_analyzer.get_flow_alerts(flow_metrics)
            
            analysis_results['flow_analysis'] = {
                'metrics': flow_metrics,
                'summary': flow_summary,
                'alerts': flow_alerts
            }
            analysis_results['alerts'].extend(flow_alerts)
        
        # Risk Analysis
        if self.risk_analysis_enabled:
            # Extract risk factors
            density = basic_metrics['density']
            flow_conflict = analysis_results['flow_analysis']['metrics'].flow_conflict_score if self.flow_analysis_enabled else 0.0
            speed_variation = analysis_results['flow_analysis']['metrics'].speed_variation_score if self.flow_analysis_enabled else 0.0
            
            # Compute risk assessment
            risk_assessment = self.risk_engine.compute_comprehensive_risk(
                density, flow_conflict, speed_variation
            )
            
            analysis_results['risk_analysis'] = risk_assessment
            analysis_results['alerts'].append(risk_assessment['risk_alert'])
        
        # Create annotated frame
        annotated_frame = self._create_annotated_frame(
            frame, tracking_results, analysis_results,
            show_zones, show_flow_arrows, show_risk_banner
        )
        analysis_results['annotated_frame'] = annotated_frame
        
        return analysis_results
    
    def _basic_processing(self, frame: np.ndarray, tracking_results: List[Dict]) -> Dict:
        """
        Basic processing when advanced analytics are disabled.
        
        Args:
            frame: Input frame
            tracking_results: List of tracking results
            
        Returns:
            Dictionary with basic analysis results
        """
        basic_metrics = self.compute_basic_metrics(tracking_results)
        
        # Simple annotated frame
        annotated_frame = frame.copy()
        
        # Draw basic info
        cv2.putText(annotated_frame, f"People: {basic_metrics['people_count']}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Density: {basic_metrics['density']:.4f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return {
            'basic_metrics': basic_metrics,
            'risk_analysis': None,
            'zone_analysis': None,
            'flow_analysis': None,
            'alerts': [],
            'annotated_frame': annotated_frame
        }
    
    def _create_annotated_frame(self, 
                              frame: np.ndarray, 
                              tracking_results: List[Dict],
                              analysis_results: Dict,
                              show_zones: bool,
                              show_flow_arrows: bool,
                              show_risk_banner: bool) -> np.ndarray:
        """
        Create comprehensive annotated frame with all overlays.
        
        Args:
            frame: Input frame
            tracking_results: List of tracking results
            analysis_results: Analysis results
            show_zones: Whether to show zone overlay
            show_flow_arrows: Whether to show flow arrows
            show_risk_banner: Whether to show risk banner
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw zone overlay
        if show_zones and self.zone_analysis_enabled:
            annotated_frame = self.zone_analyzer.draw_zones_on_frame(annotated_frame, show_labels=True)
        
        # Draw flow arrows
        if show_flow_arrows and self.flow_analysis_enabled:
            annotated_frame = self.flow_analyzer.draw_flow_arrows(annotated_frame, tracking_results)
        
        # Draw tracking IDs and bounding boxes
        for result in tracking_results:
            track_id = result.get('track_id')
            bbox = result.get('bbox')
            
            if track_id is not None and bbox:
                x1, y1, x2, y2 = bbox
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(annotated_frame, f"ID:{track_id}", 
                           (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 1)
        
        # Draw risk banner
        if show_risk_banner and self.risk_analysis_enabled and analysis_results['risk_analysis']:
            risk_data = analysis_results['risk_analysis']
            self._draw_risk_banner(annotated_frame, risk_data)
        
        # Draw metrics panel
        self._draw_metrics_panel(annotated_frame, analysis_results)
        
        # Draw alerts
        if analysis_results['alerts']:
            self._draw_alerts(annotated_frame, analysis_results['alerts'])
        
        return annotated_frame
    
    def _draw_risk_banner(self, frame: np.ndarray, risk_data: Dict):
        """Draw risk level banner on frame."""
        risk_level = risk_data['risk_level']
        risk_color = risk_data['risk_color']
        risk_score = risk_data['risk_score']
        risk_alert = risk_data['risk_alert']
        
        # Banner dimensions
        banner_height = 60
        banner_y = 10
        
        # Draw banner background
        banner_width = frame.shape[1] - 20
        cv2.rectangle(frame, (10, banner_y), (banner_width, banner_y + banner_height), 
                    risk_color, -1)
        
        # Add border
        cv2.rectangle(frame, (10, banner_y), (banner_width, banner_y + banner_height), 
                    (255, 255, 255), 2)
        
        # Draw risk level
        cv2.putText(frame, f"RISK LEVEL: {risk_level}", 
                   (20, banner_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Draw risk score
        cv2.putText(frame, f"Score: {risk_score:.2f}", 
                   (20, banner_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    def _draw_metrics_panel(self, frame: np.ndarray, analysis_results: Dict):
        """Draw metrics panel on frame."""
        panel_x = frame.shape[1] - 200
        panel_y = 80
        panel_width = 190
        panel_height = 150
        
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (255, 255, 255), 1)
        
        # Draw metrics
        y_offset = panel_y + 20
        line_height = 18
        
        # Basic metrics
        basic_metrics = analysis_results['basic_metrics']
        cv2.putText(frame, f"People: {basic_metrics['people_count']}", 
                   (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Density: {basic_metrics['density']:.3f}", 
                   (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (255, 255, 255), 1)
        y_offset += line_height
        
        # Flow metrics
        if analysis_results['flow_analysis']:
            flow_metrics = analysis_results['flow_analysis']['metrics']
            cv2.putText(frame, f"Flow: {flow_metrics.dominant_direction}", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Objects: {flow_metrics.total_objects}", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
            y_offset += line_height
        
        # Zone metrics
        if analysis_results['zone_analysis']:
            zone_summary = analysis_results['zone_analysis']['summary']
            overcrowded = len(zone_summary['overcrowded_zones'])
            cv2.putText(frame, f"Overcrowded: {overcrowded}", 
                       (panel_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
    
    def _draw_alerts(self, frame: np.ndarray, alerts: List[str]):
        """Draw alerts on frame."""
        alert_y = frame.shape[0] - 60
        
        for i, alert in enumerate(alerts[-3:]):  # Show last 3 alerts
            # Alert background
            cv2.rectangle(frame, (10, alert_y + i * 20), 
                          (frame.shape[1] - 10, alert_y + i * 20 + 18), 
                          (0, 0, 255), -1)
            
            # Alert text
            cv2.putText(frame, alert, 
                       (15, alert_y + i * 20 + 13), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1)
    
    def update_configuration(self, 
                            risk_weights: Dict[str, float] = None,
                            grid_size: Tuple[int, int] = None,
                            restricted_zones: List[str] = None,
                            enabled_modules: Dict[str, bool] = None):
        """
        Update analytics configuration.
        
        Args:
            risk_weights: New risk weights
            grid_size: New zone grid size
            restricted_zones: New restricted zones
            enabled_modules: Module enable/disable flags
        """
        # Update risk weights
        if risk_weights:
            self.risk_engine.update_weights(
                risk_weights.get('density'),
                risk_weights.get('flow_conflict'),
                risk_weights.get('speed_variation')
            )
        
        # Update grid size
        if grid_size:
            self.zone_analyzer.update_grid_size(grid_size)
        
        # Update restricted zones
        if restricted_zones is not None:
            self.zone_analyzer.set_restricted_zones(restricted_zones)
        
        # Update module states
        if enabled_modules:
            self.analytics_enabled = enabled_modules.get('analytics', True)
            self.zone_analysis_enabled = enabled_modules.get('zones', True)
            self.flow_analysis_enabled = enabled_modules.get('flow', True)
            self.risk_analysis_enabled = enabled_modules.get('risk', True)
    
    def get_comprehensive_summary(self, analysis_results: Dict) -> Dict:
        """
        Get comprehensive analysis summary.
        
        Args:
            analysis_results: Analysis results from process_frame_advanced
            
        Returns:
            Comprehensive summary dictionary
        """
        summary = {
            'timestamp': np.datetime64('now').astype(int),
            'basic_metrics': analysis_results['basic_metrics'],
            'alerts': analysis_results['alerts'],
            'module_status': {
                'risk_analysis': self.risk_analysis_enabled,
                'zone_analysis': self.zone_analysis_enabled,
                'flow_analysis': self.flow_analysis_enabled
            }
        }
        
        # Add risk summary
        if analysis_results['risk_analysis']:
            summary['risk_summary'] = {
                'level': analysis_results['risk_analysis']['risk_level'],
                'score': analysis_results['risk_analysis']['risk_score'],
                'alert': analysis_results['risk_analysis']['risk_alert']
            }
        
        # Add zone summary
        if analysis_results['zone_analysis']:
            summary['zone_summary'] = analysis_results['zone_analysis']['summary']
        
        # Add flow summary
        if analysis_results['flow_analysis']:
            summary['flow_summary'] = analysis_results['flow_analysis']['summary']
        
        return summary
