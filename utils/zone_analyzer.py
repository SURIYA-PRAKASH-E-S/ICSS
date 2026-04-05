"""
Zone-Based Monitoring for Intelligent Crowd Analytics
Divide frame into configurable grid zones for localized analysis
Features:
- Real-world area calculation
- Zone grid with per-zone density + risk
- User-marked ROI region
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

def calculate_crowd_bounding_area(tracked_objects, scale_x, scale_y):
    """
    Given all tracked persons, find the bounding box that encloses
    ALL detected people, then convert to real-world area in m².

    tracked_objects: list of {'bbox': [x1,y1,x2,y2], 'track_id': int}
    scale_x: meters per pixel (X axis)
    scale_y: meters per pixel (Y axis)

    Returns:
        crowd_bbox_pixels: (x1, y1, x2, y2) — pixel bounding box of crowd
        crowd_area_m2: float — real-world area in square meters
        crowd_width_m: float
        crowd_height_m: float
    """
    if not tracked_objects:
        return None, 0.0, 0.0, 0.0

    all_x1 = [obj['bbox'][0] for obj in tracked_objects]
    all_y1 = [obj['bbox'][1] for obj in tracked_objects]
    all_x2 = [obj['bbox'][2] for obj in tracked_objects]
    all_y2 = [obj['bbox'][3] for obj in tracked_objects]

    crowd_x1 = min(all_x1)
    crowd_y1 = min(all_y1)
    crowd_x2 = max(all_x2)
    crowd_y2 = max(all_y2)

    pixel_w = crowd_x2 - crowd_x1
    pixel_h = crowd_y2 - crowd_y1

    crowd_width_m  = pixel_w * scale_x
    crowd_height_m = pixel_h * scale_y
    crowd_area_m2  = crowd_width_m * crowd_height_m

    return (crowd_x1, crowd_y1, crowd_x2, crowd_y2), crowd_area_m2, crowd_width_m, crowd_height_m


def draw_crowd_area(frame, crowd_bbox, crowd_area_m2, crowd_width_m, crowd_height_m):
    """Draw crowd bounding box on frame with area information."""
    if crowd_bbox is None:
        return frame
    x1, y1, x2, y2 = crowd_bbox
    # Draw dashed yellow bounding box around entire crowd
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = f"Crowd Area: {crowd_area_m2:.1f} m²  ({crowd_width_m:.1f}m x {crowd_height_m:.1f}m)"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


class ZoneAnalyzer:
    """
    Zone-based crowd monitoring system with configurable grid layout,
    real-world area calculation, and localized density analysis.
    """

    def __init__(self, frame_width, frame_height,
                 grid_rows=3, grid_cols=3,
                 real_world_width_m=50.0,
                 real_world_height_m=30.0,
                 max_density_threshold=0.5,
                 zone_count_limit=10):

        self.frame_w = frame_width
        self.frame_h = frame_height
        self.rows = grid_rows
        self.cols = grid_cols
        self.rw_width  = real_world_width_m
        self.rw_height = real_world_height_m
        self.max_density = max_density_threshold
        self.count_limit = zone_count_limit

        # Pixel-to-meter scale
        self.scale_x = real_world_width_m  / frame_width
        self.scale_y = real_world_height_m / frame_height

        self.zones = []
        self.frame_counter = 0
        self._build_zones()

    def _build_zones(self):
        """Build zone grid based on frame dimensions and real-world area."""
        self.zones = []
        zone_w = self.frame_w // self.cols
        zone_h = self.frame_h // self.rows
        zone_area_m2 = (self.rw_width * self.rw_height) / (self.rows * self.cols)

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = c * zone_w
                y1 = r * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h
                zone_id = f"Z{r * self.cols + c + 1}"
                self.zones.append({
                    'zone_id'       : zone_id,
                    'pixel_bbox'    : (x1, y1, x2, y2),
                    'real_area_m2'  : zone_area_m2,
                    'people_count'  : 0,
                    'density'       : 0.0,
                    'risk_level'    : 'LOW',
                    'is_overcrowded': False,
                    'track_ids'     : []
                })

    def assign_people_to_zones(self, tracked_objects):
        """
        Assign tracked people to zones based on center point of bounding box.
        Recalculate density and risk for each zone.
        """
        # Reset counts
        for z in self.zones:
            z['people_count'] = 0
            z['track_ids'] = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            cx = (x1 + x2) // 2   # person center X
            cy = (y1 + y2) // 2   # person center Y

            for z in self.zones:
                zx1, zy1, zx2, zy2 = z['pixel_bbox']
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    z['people_count'] += 1
                    z['track_ids'].append(obj['track_id'])
                    break

        # Recalculate density and risk for each zone
        for z in self.zones:
            count = z['people_count']
            area  = z['real_area_m2']
            density = count / area if area > 0 else 0.0
            z['density'] = round(density, 4)

            if density > self.max_density or count > self.count_limit:
                z['risk_level']    = 'HIGH'
                z['is_overcrowded'] = True
            elif density > self.max_density * 0.4 or count > self.count_limit * 0.5:
                z['risk_level']    = 'MEDIUM'
                z['is_overcrowded'] = False
            else:
                z['risk_level']    = 'LOW'
                z['is_overcrowded'] = False

    def draw_zone_grid(self, frame):
        """Draw zone grid on frame with color-coded risk levels and statistics."""
        self.frame_counter += 1
        blink = (self.frame_counter % 30) < 15  # blink every 15 frames

        for z in self.zones:
            x1, y1, x2, y2 = z['pixel_bbox']
            risk  = z['risk_level']
            count = z['people_count']
            dens  = z['density']

            # Zone color by risk
            if risk == 'HIGH':
                color = (0, 0, 255)       # Red
                alpha = 0.25
            elif risk == 'MEDIUM':
                color = (0, 165, 255)     # Orange
                alpha = 0.15
            else:
                color = (0, 255, 0)       # Green
                alpha = 0.08

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Border — blink red if overcrowded
            border_color = (0, 0, 255) if (z['is_overcrowded'] and blink) else color
            border_thick = 3 if z['is_overcrowded'] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

            # Zone ID label
            cv2.putText(frame, z['zone_id'], (x1 + 6, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Stats inside zone
            cv2.putText(frame, f"People: {count}", (x1 + 6, y1 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{dens:.3f} p/m2", (x1 + 6, y1 + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, risk, (x1 + 6, y1 + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Alert text
            if z['is_overcrowded'] and blink:
                cv2.putText(frame, "!! ALERT", (x1 + 6, y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def get_zone_summary(self):
        """Return list of zone dictionaries with current statistics."""
        return self.zones

    def pixel_to_realworld(self, x_pixel, y_pixel):
        """Convert pixel coordinates to real-world coordinates in meters."""
        return round(x_pixel * self.scale_x, 2), round(y_pixel * self.scale_y, 2)

    def calculate_roi_stats(self, tracked_objects, roi_bbox):
        """
        Calculate statistics for a user-defined ROI region.

        roi_bbox: (x1, y1, x2, y2) in pixels — user defined region
        Returns people inside ROI, ROI area in m², density, risk
        """
        rx1, ry1, rx2, ry2 = roi_bbox
        roi_pixel_w = rx2 - rx1
        roi_pixel_h = ry2 - ry1
        roi_area_m2 = (roi_pixel_w * self.scale_x) * (roi_pixel_h * self.scale_y)

        people_in_roi = []
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                people_in_roi.append(obj['track_id'])

        count    = len(people_in_roi)
        density  = count / roi_area_m2 if roi_area_m2 > 0 else 0.0

        if density > self.max_density:
            risk = 'HIGH'
        elif density > self.max_density * 0.4:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return {
            'count'      : count,
            'area_m2'    : round(roi_area_m2, 2),
            'density'    : round(density, 4),
            'risk'       : risk,
            'track_ids'  : people_in_roi
        }

    def draw_roi(self, frame, roi_bbox, roi_stats):
        """Draw ROI region on frame with statistics."""
        rx1, ry1, rx2, ry2 = roi_bbox
        color = {'HIGH': (0,0,255), 'MEDIUM': (0,165,255), 'LOW': (0,255,0)}
        c = color[roi_stats['risk']]
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), c, 3)
        cv2.putText(frame, f"ROI | {roi_stats['count']} people",
                    (rx1, ry1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        cv2.putText(frame, f"{roi_stats['area_m2']} m² | {roi_stats['density']:.3f} p/m²",
                    (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        return frame

    def check_alerts(self):
        """Returns list of overcrowded zone alert messages."""
        alerts = []
        for z in self.zones:
            if z['is_overcrowded']:
                alerts.append({
                    'zone_id'   : z['zone_id'],
                    'count'     : z['people_count'],
                    'density'   : z['density'],
                    'risk'      : z['risk_level'],
                    'message'   : f"⚠️ {z['zone_id']} OVERCROWDED: "
                                  f"{z['people_count']} people, "
                                  f"{z['density']:.3f} p/m²"
                })
        return alerts

    def update_frame_size(self, new_width, new_height):
        """Update frame dimensions and recreate zones."""
        self.frame_w = new_width
        self.frame_h = new_height
        self.scale_x = self.rw_width / new_width
        self.scale_y = self.rw_height / new_height
        self._build_zones()

    def update_grid_config(self, grid_rows=None, grid_cols=None,
                          real_world_width_m=None, real_world_height_m=None,
                          max_density_threshold=None, zone_count_limit=None):
        """Update zone grid configuration and rebuild zones."""
        if grid_rows is not None:
            self.rows = grid_rows
        if grid_cols is not None:
            self.cols = grid_cols
        if real_world_width_m is not None:
            self.rw_width = real_world_width_m
        if real_world_height_m is not None:
            self.rw_height = real_world_height_m
        if max_density_threshold is not None:
            self.max_density = max_density_threshold
        if zone_count_limit is not None:
            self.count_limit = zone_count_limit

        # Recalculate scale factors
        self.scale_x = self.rw_width / self.frame_w
        self.scale_y = self.rw_height / self.frame_h
        self._build_zones()
