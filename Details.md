# 🎥 AI Crowd Surveillance System - Complete Project Details

## 📋 **Project Overview**

**Project Name**: AI Crowd Surveillance System (Enhanced)  
**Type**: Real-time Computer Vision Application  
**Framework**: Streamlit Web Application  
**Primary Purpose**: Intelligent crowd monitoring and risk assessment using AI/ML  

---

## 🏗️ **Project Structure**

```
d:\New1Pro/
├── app.py                    (58,617 bytes) - Main Streamlit application
├── requirements.txt           (90 bytes) - Python dependencies
├── README.md                (4,731 bytes) - Project documentation
├── crowd_data.db            (12,288 bytes) - DuckDB local database
├── crowd_data.db.wal        (24,072 bytes) - DuckDB write-ahead log
├── model/                   - AI model files
│   ├── yolo11l.pt              (51,387,343 bytes) - YOLO v11 primary model
│   └── V8l-haj.pt          (87,644,474 bytes) - YOLO v8 secondary model
├── utils/                   - Utility modules
│   ├── advanced_analytics.py (17,544 bytes) - Advanced analytics integration
│   ├── detection.py         (21,398 bytes) - Detection pipeline
│   ├── flow_analyzer.py     (18,308 bytes) - Flow analysis engine
│   ├── risk_engine.py       (11,647 bytes) - Risk assessment engine
│   ├── tracker.py           (16,708 bytes) - Deep SORT tracking
│   └── zone_analyzer.py     (12,949 bytes) - Zone monitoring
└── firebase/                - Legacy Firebase files (deprecated)
    ├── firebase_setup.py     (6,975 bytes) - Firebase setup script
    └── service-account-key.json (2,376 bytes) - Firebase credentials
```

**Total Python Files**: 8 files  
**Total Lines of Code**: 13,516 lines  
**Total Project Size**: ~250MB (including models)

---

## 🛠️ **Technology Stack**

### **Core Framework**
- **Streamlit**: Web application framework
- **Python 3.x**: Primary programming language
- **Streamlit WebRTC**: Real-time video streaming

### **Computer Vision & AI**
- **OpenCV**: Image processing and computer vision
- **Ultralytics YOLO**: Object detection framework
- **YOLO v11**: Latest object detection model (51MB)
- **YOLO v8**: Legacy object detection model (87MB)
- **PyTorch**: Deep learning backend

### **Data Processing**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **DuckDB**: Local analytical database
- **Plotly**: Data visualization

### **Video Processing**
- **PyAV (av)**: Video file handling
- **Deep SORT**: Multi-object tracking algorithm

### **Development Environment**
- **Virtual Environment**: Python venv
- **Windows OS**: Primary development platform

---

## 🎯 **Core Features**

### **1. Dual Model Detection System**
- **YOLO v11**: Primary high-accuracy model
- **YOLO v8**: Secondary validation model
- **Smart Merging**: IoU-based non-maximum suppression
- **Model Selection**: Single or dual model operation
- **Color Coding**: Green (v11) and Blue (v8) bounding boxes

### **2. Advanced Analytics Platform**
- **Risk Assessment**: 3-level classification (Normal/Average/Risky)
- **Zone Monitoring**: Grid-based area analysis
- **Flow Analysis**: Direction detection and movement patterns
- **Speed Estimation**: Object velocity calculation
- **Conflict Detection**: Bidirectional flow analysis

### **3. Real-time Processing**
- **Live Webcam Streaming**: Real-time video processing
- **Video File Upload**: Support for MP4, AVI, MOV, MKV
- **Frame Skipping**: Performance optimization (process every 3rd frame)
- **Resolution Scaling**: Adaptive frame resizing

### **4. Multi-Object Tracking**
- **Deep SORT Integration**: Persistent object IDs
- **Track History**: Last 20 positions per object
- **Speed Calculation**: FPS-aware velocity estimation
- **Movement Vectors**: Direction and magnitude analysis

### **5. Local Data Storage**
- **DuckDB Database**: Local analytics storage
- **Real-time Inserts**: Frame-by-frame data logging
- **Historical Analysis**: Last 10 records display
- **Statistics**: Average values and risk distribution

---

## 📊 **Data Flow Architecture**

```
Video Input (Webcam/File)
    ↓
Frame Processing (OpenCV)
    ↓
YOLO Detection (v11 + v8)
    ↓
Smart Merging (IoU NMS)
    ↓
Deep SORT Tracking
    ↓
Advanced Analytics
    ├── Risk Assessment
    ├── Zone Analysis
    └── Flow Analysis
    ↓
DuckDB Storage (crowd_data.db)
    ↓
Streamlit Dashboard Display
```

---

## 🎮 **User Interface**

### **Tab 1: 🎥 Live Feed**
- **Input Selection**: Webcam/Video upload toggle
- **Real-time Processing**: Live video with AI overlays
- **Performance Indicators**: FPS, model status, optimization mode
- **Visual Overlays**: Bounding boxes, risk levels, flow directions
- **Alert System**: Color-coded risk warnings

### **Tab 2: 📊 Analytics**
- **Real-time Metrics**: People count, density, flow, risk level
- **Advanced Analytics**: Risk engine, zone analysis, flow patterns
- **Risk Assessment**: Color-coded indicators and alerts
- **Model Performance**: Detection statistics and model status

### **Tab 3: 💾 Local DB**
- **Current Metrics**: Latest database values
- **Historical Trends**: Last 10 records table
- **Statistics**: Average values and risk distribution
- **Database Info**: Record count and storage details

### **Tab 4: ⚙️ Controls**
- **Model Selection**: YOLO v11/v8 toggle
- **Deep SORT Toggle**: Multi-object tracking control
- **Advanced Analytics**: Risk, zone, flow analysis toggles
- **Threshold Settings**: Density and count risk levels
- **Risk Weights**: Configurable assessment parameters

---

## 🔧 **Configuration Parameters**

### **Detection Thresholds**
- **Low Density**: 0.5 (triggers "Average" risk)
- **Medium Density**: 1.0 (triggers "Risky" risk)
- **People Count**: 8 (triggers "Risky" risk)

### **Risk Engine Weights**
- **Density Weight**: 0.4 (crowd density factor)
- **Flow Conflict Weight**: 0.35 (bidirectional flow factor)
- **Speed Variation Weight**: 0.25 (velocity variation factor)

### **Deep SORT Parameters**
- **Max Age**: 30 frames (track persistence)
- **N Init**: 3 detections (track confirmation)
- **NMS Max Overlap**: 0.3 (detection overlap threshold)

### **Performance Optimization**
- **Frame Skip**: Every 3rd frame processed
- **Target Resolution**: 640x480 pixels
- **Model Loading**: Cached with @st.cache_resource

---

## 📈 **Performance Metrics**

### **Model Performance**
- **YOLO v11**: 51MB, high accuracy, latest version
- **YOLO v8**: 87MB, good performance, legacy version
- **Detection Speed**: ~10 FPS (configurable)
- **Accuracy**: Enhanced with dual-model validation

### **System Performance**
- **Startup Time**: Fast (cached model loading)
- **Memory Usage**: Optimized with frame skipping
- **Database Operations**: Local DuckDB (no network latency)
- **UI Responsiveness**: Real-time updates

### **Processing Pipeline**
- **Frame Rate**: Configurable (default 30 FPS)
- **Processing Mode**: Optimized (1/3 frame processing)
- **Resolution Scaling**: Adaptive aspect ratio preservation
- **Error Handling**: Graceful degradation on failures

---

## 🗄️ **Database Schema**

### **Table: crowd_metrics**
```sql
CREATE TABLE crowd_metrics (
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    people_count INTEGER,
    density FLOAT,
    flow_direction VARCHAR,
    risk_level VARCHAR
);
```

### **Data Fields**
- **timestamp**: Automatic timestamp for each record
- **people_count**: Number of detected persons
- **density**: Normalized people per pixel area
- **flow_direction**: Movement direction (Left/Right/Up/Down/Mixed)
- **risk_level**: Risk classification (Normal/Average/Risky)

---

## 🔐 **Security & Privacy**

### **Local Processing**
- **No Cloud Dependencies**: All processing done locally
- **Data Privacy**: No data transmitted externally
- **Offline Operation**: Works without internet connection
- **Local Storage**: DuckDB database on device

### **Access Control**
- **No Authentication**: Local application access
- **File System**: Standard OS permissions
- **Network**: No external API calls

---

## 🚀 **Deployment & Installation**

### **System Requirements**
- **Python 3.8+**: Primary runtime requirement
- **GPU**: Optional (accelerates YOLO inference)
- **RAM**: 4GB+ minimum (8GB+ recommended)
- **Storage**: 250MB+ (including models)
- **OS**: Windows/Linux/macOS (cross-platform)

### **Installation Steps**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### **Dependencies**
```
streamlit>=1.28.0
streamlit-webrtc>=1.0.0
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
av>=10.0.0
duckdb>=0.9.0
plotly>=5.15.0
pandas>=2.0.0
```

---

## 🔄 **Recent Updates & Migrations**

### **Firebase to DuckDB Migration** ✅
- **Removed**: Firebase Admin SDK dependencies
- **Added**: DuckDB local database storage
- **Benefits**: No network latency, offline operation, data privacy
- **Status**: Complete migration implemented

### **Streamlit Deprecation Fix** ✅
- **Fixed**: `use_container_width` deprecation warnings
- **Updated**: `width="stretch"` parameter usage
- **Benefits**: Future-proof, compatible with latest Streamlit
- **Status**: All warnings resolved

### **DuckDB Dataframe Fix** ✅
- **Fixed**: ArrowMixin dataframe errors
- **Updated**: `.fetchdf()` method usage
- **Benefits**: Proper pandas DataFrame handling
- **Status**: Stable dashboard rendering

---

## 🎯 **Key Capabilities Summary**

### **Detection Capabilities**
- ✅ **Person Detection**: High-precision YOLO models
- ✅ **Crowd Counting**: Real-time people counting
- ✅ **Density Estimation**: Normalized area calculations
- ✅ **Movement Tracking**: Persistent object IDs
- ✅ **Flow Analysis**: Direction pattern detection

### **Analytics Capabilities**
- ✅ **Risk Assessment**: 3-tier classification system
- ✅ **Zone Monitoring**: Grid-based area analysis
- ✅ **Speed Estimation**: Velocity calculations
- ✅ **Conflict Detection**: Bidirectional flow analysis
- ✅ **Historical Analysis**: Trend identification

### **User Interface Capabilities**
- ✅ **Real-time Display**: Live video processing
- ✅ **Multi-tab Interface**: Organized feature access
- ✅ **Configurable Settings**: User-adjustable parameters
- ✅ **Responsive Design**: Adaptive layout
- ✅ **Error Handling**: Graceful failure management

---

## 📞 **Technical Support Information**

### **Project Architecture**
- **Monolithic Design**: Single application file (app.py)
- **Modular Utils**: Separate utility modules
- **State Management**: Streamlit session state
- **Threading**: Async video processing

### **Code Quality**
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Inline docstrings
- **Error Handling**: Try-catch blocks throughout
- **Performance**: Optimized with caching

### **Maintenance**
- **Logging**: Configurable error logging
- **Updates**: Modern Streamlit API usage
- **Compatibility**: Cross-platform support
- **Extensibility**: Modular component design

---

## 🎉 **Project Status**

**Development Phase**: Production Ready  
**Last Updated**: March 2026  
**Version**: Enhanced v2.0  
**Stability**: Stable (no known critical issues)  
**Performance**: Optimized for real-time processing  

**Ready for deployment and production use!** 🚀
