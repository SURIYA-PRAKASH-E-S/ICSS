# AI Crowd Surveillance System - Complete Project Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [File Structure](#file-structure)
7. [Core Components](#core-components)
8. [API Reference](#api-reference)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)
11. [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

The AI Crowd Surveillance System is a production-ready application that combines multiple computer vision techniques for real-time crowd monitoring and analysis. It leverages YOLO models for detection, Deep SORT for tracking, and advanced analytics for crowd behavior analysis.

### Key Capabilities
- **Real-time Detection**: Multi-model YOLO detection (v11, v8, v11m)
- **Persistent Tracking**: Deep SORT multi-object tracking with unique IDs
- **Crowd Analytics**: Density estimation, flow analysis, risk assessment
- **Interactive Dashboard**: Streamlit-based UI with live analytics
- **Data Storage**: DuckDB for real-time metrics storage

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Source   │───▶│  Video Processor│───▶│   Detection     │
│  (Webcam/Video) │    │   (Frame Skip)  │    │    (YOLO)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│   Analytics     │◀───│   Tracking      │
│  (Dashboard)    │    │   (Crowd Data)  │    │  (Deep SORT)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DuckDB Store  │◀───│   Visualization │◀───│   Risk Engine   │
│  (Metrics DB)   │    │  (Heatmaps)     │    │ (Classification)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## ✨ Features

### 🔍 Detection Features
- **Multi-Model Support**: YOLOv11 Large, YOLOv8 Large, YOLOv11 Medium
- **Person-Only Detection**: Filtered for person class (class 0)
- **IoU-based NMS**: Merges detections from multiple models
- **Confidence Thresholding**: Configurable confidence levels

### 🎯 Tracking Features
- **Deep SORT Integration**: Persistent multi-object tracking
- **Unique IDs**: Each person gets a unique tracking ID
- **Speed & Direction**: Real-time movement analysis
- **Trajectory History**: Visual tracking paths

### 📊 Analytics Features
- **Density Calculation**: `density = (count / (width × height)) × 10000`
- **Crowd Levels**: Low/Medium/High classification
- **Risk Assessment**: Multi-factor risk evaluation
- **Flow Analysis**: Movement direction patterns
- **Heatmaps**: Density and zone-based visualization

### 🎨 UI Features
- **Live Webcam Feed**: Real-time processing
- **Video Upload**: Frame-by-frame analysis
- **Interactive Controls**: Model selection, thresholds
- **Analytics Dashboard**: Real-time metrics display
- **Clean Interface**: Minimal overlays, focused visualization

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
OpenCV 4.5+
CUDA (optional, for GPU acceleration)
```

### Setup Steps
1. **Clone Repository**
```bash
git clone <repository-url>
cd New2crowd
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download Models**
Place the following models in the `model/` directory:
- `yolo11l.pt` - YOLOv11 Large model
- `V8l-haj.pt` - YOLOv8 Large model  
- `yolov11m.pt` - YOLOv11 Medium model

5. **Run Application**
```bash
streamlit run app.py
```

---

## ⚙️ Configuration

### Model Selection
Access via **⚙️ Controls** tab:
- Enable/disable individual YOLO models
- Active models are combined for better detection

### Threshold Settings
- **Low Crowd Count**: Default 5 people
- **Medium Crowd Count**: Default 15 people
- **Low Density Threshold**: Default 0.3
- **Medium Density Threshold**: Default 0.7

### Performance Settings
- **Frame Skipping**: Process every 3rd frame (default)
- **Target Resolution**: 640×480 for faster inference
- **FPS Target**: 25-30 fps

---

## 📁 File Structure

```
New2crowd/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── crowd_data.db            # DuckDB database (auto-created)
├── model/                   # Model files directory
│   ├── yolo11l.pt          # YOLOv11 Large
│   ├── V8l-haj.pt          # YOLOv8 Large
│   └── yolov11m.pt         # YOLOv11 Medium
├── utils/                   # Utility modules
│   ├── detection.py        # Detection pipeline
│   ├── tracker.py          # Deep SORT tracking
│   ├── crowd_analytics.py  # Crowd analytics
│   ├── crowd_visualization.py # Visualization
│   ├── advanced_analytics.py # Advanced analytics
│   ├── csrnet_density.py   # CSRNet density estimation
│   ├── integrated_detection.py # Integrated pipeline
│   ├── risk_engine.py      # Risk assessment
│   ├── flow_analyzer.py    # Flow analysis
│   └── zone_analyzer.py    # Zone analysis
└── README.md               # Project documentation
```

---

## 🔧 Core Components

### 1. VideoProcessor Class (`app.py`)
Handles real-time video processing with frame skipping and resizing.

**Key Methods:**
- `recv()`: Main processing pipeline
- `resize_frame()`: Aspect-ratio preserving resize

### 2. Detection Pipeline (`utils/detection.py`)
Multi-model YOLO detection with IoU-based merging.

**Key Functions:**
- `process_frame_with_deep_sort()`: Main detection + tracking pipeline
- `merge_detections()`: IoU-based NMS for multi-model results
- `calculate_density()`: Crowd density calculation
- `classify_risk()`: Risk level classification

### 3. Deep SORT Tracker (`utils/tracker.py`)
Persistent multi-object tracking with unique IDs.

**Key Methods:**
- `update_tracks()`: Update tracker with new detections
- `draw_tracking_info()`: Draw bounding boxes and IDs
- `calculate_speed()`: Speed estimation from position history

### 4. Crowd Analytics (`utils/crowd_analytics.py`)
Advanced crowd behavior analysis.

**Features:**
- Peak count tracking
- Average count calculation
- Crowd level classification
- Heatmap generation

### 5. Visualization (`utils/crowd_visualization.py`)
Drawing utilities for overlays and visualizations.

**Functions:**
- `draw_crowd_overlay()`: Clean UI overlay
- `draw_heatmap_overlay()`: Density heatmap
- `draw_zone_grid()`: Zone-based analysis grid

---

## 📖 API Reference

### Detection Functions

#### `process_frame_with_deep_sort(frame, model_v11, model_v8, tracker, active_models, fps, model_v11m=None)`
Process frame with YOLO detection + Deep SORT tracking.

**Parameters:**
- `frame`: Input video frame (numpy array)
- `model_v11`: YOLOv11 model instance
- `model_v8`: YOLOv8 model instance
- `model_v11m`: YOLOv11m model instance (optional)
- `tracker`: Deep SORT tracker instance
- `active_models`: List of active model names
- `fps`: Frames per second for speed calculation

**Returns:**
```python
{
    'annotated_frame': frame_with_overlays,
    'people_count': int,
    'density': float,
    'flow_direction': str,
    'risk_level': str,
    'risk_color': tuple,
    'model_info': dict,
    'avg_speed': float,
    'tracked_objects': list,
    'tracking_results': dict
}
```

#### `calculate_density(count, width, height)`
Calculate normalized crowd density.

**Formula:** `density = (count / (width × height)) × 10000`

#### `classify_risk(density, count, low_threshold, medium_threshold, count_threshold)`
Classify risk level based on density and count.

**Returns:** `(risk_level, risk_color)`

### Tracker Methods

#### `ObjectTracker.update_tracks(detections, confidences)`
Update Deep SORT tracker with new detections.

#### `ObjectTracker.draw_tracking_info(frame, tracking_results)`
Draw bounding boxes, IDs, and trajectories.

### Analytics Functions

#### `init_crowd_analytics(low_count_threshold, medium_count_threshold, low_density_threshold, medium_density_threshold)`
Initialize crowd analytics with thresholds.

#### `update_metrics(people_count, density, tracked_objects)`
Update crowd metrics and return analytics data.

---

## ⚡ Performance Optimization

### Frame Processing
- **Frame Skipping**: Process every 3rd frame for better performance
- **Resolution Reduction**: 640×480 target size for faster inference
- **Async Processing**: Thread-safe session state management

### Model Optimization
- **Model Selection**: Enable only needed models
- **Silent Inference**: `verbose=False` to reduce console output
- **GPU Acceleration**: Automatic CUDA detection if available

### Memory Management
- **Frame Caching**: Store last processed frame for skipping
- **History Limits**: Limited track history (20 positions)
- **Resource Cleanup**: Proper file handle management

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Missing ScriptRunContext" Warning
**Cause**: Normal warning in async processing
**Solution**: Can be safely ignored

#### 2. Model Not Found
**Cause**: Missing model files in `model/` directory
**Solution**: Download required models:
- `yolo11l.pt`
- `V8l-haj.pt`
- `yolov11m.pt`

#### 3. Low FPS
**Causes**:
- Multiple active models
- High resolution input
- CPU-only processing

**Solutions**:
- Use single model
- Ensure models are in `model/` directory
- Check if GPU is available

#### 4. No Bounding Boxes
**Causes**:
- Active models not selected
- Low confidence threshold
- No persons in frame

**Solutions**:
- Go to **⚙️ Controls** tab
- Select at least one model
- Ensure proper lighting and subjects

#### 5. Database Errors
**Cause**: DuckDB file permission issues
**Solution**: Delete `crowd_data.db` and restart

### Debug Mode
Add debug prints by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🚀 Future Enhancements

### Planned Features
1. **Alert System**: Real-time notifications for crowd anomalies
2. **Historical Analytics**: Long-term trend analysis
3. **Export Features**: CSV/JSON data export
4. **Camera Management**: Multiple camera support
5. **Cloud Storage**: Remote data synchronization
6. **Mobile App**: Cross-platform mobile application

### Model Improvements
1. **Custom Training**: Domain-specific model fine-tuning
2. **Ensemble Methods**: Advanced model combination
3. **Edge Deployment**: Optimized models for edge devices
4. **Real-time Alerts**: SMS/email notifications

### UI Enhancements
1. **Dark Mode**: Theme switching
2. **Custom Layouts**: Configurable dashboard
3. **Multi-language**: Internationalization support
4. **Accessibility**: Screen reader support

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Last Updated**: April 2026
**Version**: 2.0
**Framework**: Streamlit + OpenCV + YOLO + Deep SORT
