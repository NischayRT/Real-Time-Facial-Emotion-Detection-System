# 🎭 Advanced Real-Time Emotion Detection System v2.0

A state-of-the-art deep learning system for real-time facial emotion recognition with multiple improvements over the basic version. The system uses transfer learning, temporal smoothing, and advanced face detection to classify emotions with high accuracy.

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🆕 What's New in v2.0

### Major Improvements

✨ **Enhanced Face Detection**
- MTCNN support for better face detection at multiple angles
- Fallback to Haar Cascade for compatibility
- Higher detection accuracy and robustness

✨ **Temporal Smoothing**
- Reduces prediction jitter across frames
- More stable and natural emotion transitions
- Configurable smoothing window

✨ **Confidence Thresholding**
- Only displays predictions above confidence threshold
- Shows "Uncertain" for low-confidence predictions
- Prevents false positives

✨ **Real-Time Analytics**
- Live FPS counter
- Face count tracking
- Frame counting

✨ **Emotion Logging**
- Automatic CSV logging of all detections
- Timestamp, emotion, confidence tracking
- Full probability distribution logging

✨ **Enhanced Visualization**
- Color-coded emotion labels
- Optional probability bars for all emotions
- Multi-face support with individual tracking
- Screenshot capture functionality

✨ **Improved Training**
- Multiple architecture support (MobileNet, EfficientNet, ResNet)
- Enhanced data augmentation
- Comprehensive callbacks and monitoring
- TensorBoard integration
- Training history visualization
- Automatic best model selection

✨ **Better Model Architecture**
- Batch normalization layers
- Optimized dropout rates
- Better regularization
- Improved convergence

---

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Training Guide](#training-guide)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [API Documentation](#api-documentation)
- [Comparison: v1.0 vs v2.0](#comparison-v10-vs-v20)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)

---

## ✨ Features

### Core Features
- **Real-time emotion detection** from webcam with optimized performance
- **5 emotion categories**: Angry, Happy, Neutral, Sad, Surprise
- **Multiple model architectures**: Choose between MobileNet, EfficientNet, or ResNet
- **Advanced face detection**: MTCNN or Haar Cascade
- **Temporal smoothing**: Stable predictions across frames
- **Multi-face support**: Detect emotions for multiple people simultaneously

### User Experience
- **Interactive controls**: Toggle features on/off in real-time
- **Visual feedback**: Color-coded labels and probability bars
- **Performance monitoring**: Live FPS display
- **Screenshot capture**: Save frames with detections
- **Confidence display**: See prediction certainty

### Data & Logging
- **Automatic logging**: CSV export of all detections
- **Training history**: Plots and metrics visualization
- **TensorBoard support**: Deep training insights
- **Model checkpointing**: Save best performing models

### Performance
- **Optimized inference**: 20-30 FPS on modern hardware
- **GPU acceleration**: Full CUDA support
- **Mixed precision training**: Faster training on compatible GPUs
- **Efficient data pipeline**: Optimized batch processing

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ADVANCED PIPELINE v2.0                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Webcam/Video Feed                                       │
│         ↓                                                        │
│  FACE DETECTION: MTCNN/Haar Cascade                            │
│         ↓                                                        │
│  PREPROCESSING:                                                  │
│    - Face alignment                                             │
│    - Resize to 224×224                                          │
│    - Normalization [0, 1]                                       │
│         ↓                                                        │
│  FEATURE EXTRACTION:                                            │
│    - Base: MobileNet/EfficientNet/ResNet                       │
│    - Global Average Pooling                                     │
│         ↓                                                        │
│  CLASSIFICATION HEAD:                                           │
│    - Dense(1024) + BatchNorm + Dropout(0.5)                    │
│    - Dense(512) + BatchNorm + Dropout(0.3)                     │
│    - Dense(256) + Dropout(0.2)                                 │
│    - Dense(5, softmax)                                          │
│         ↓                                                        │
│  TEMPORAL SMOOTHING:                                            │
│    - Rolling average over 7 frames                             │
│    - Reduces jitter                                             │
│         ↓                                                        │
│  CONFIDENCE THRESHOLDING:                                       │
│    - Display only if confidence > 0.5                          │
│    - Mark uncertain predictions                                 │
│         ↓                                                        │
│  OUTPUT:                                                        │
│    - Visual display with labels                                 │
│    - CSV logging                                                │
│    - Performance metrics                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended for training)
- **GPU**: Optional but recommended (NVIDIA GPU with CUDA 11.2+ for training)
- **Webcam**: Required for real-time detection (720p or better recommended)
- **Storage**: 10GB free space (for datasets and models)
- **Python**: 3.7 or higher (3.8-3.10 recommended)

### Software Dependencies

#### Core Libraries
```
Python >= 3.7
TensorFlow >= 2.8.0
OpenCV >= 4.5.0
NumPy >= 1.19.0
```

#### Optional but Recommended
```
MTCNN >= 0.1.1 (improved face detection)
MediaPipe >= 0.8.0 (alternative face detection)
Matplotlib >= 3.3.0 (training visualization)
```

---

## 🚀 Installation

### Step 1: Clone or Download

```bash
# Clone repository
git clone https://github.com/yourusername/emotion-detection-v2.git
cd emotion-detection-v2

# Or download and extract ZIP
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv emotion_env
emotion_env\Scripts\activate

# macOS/Linux
python3 -m venv emotion_env
source emotion_env/bin/activate
```

### Step 3: Install Dependencies

#### Basic Installation (without MTCNN)
```bash
pip install -r requirements_advanced.txt
```

#### Full Installation (with MTCNN for better face detection)
```bash
pip install -r requirements_advanced.txt
pip install mtcnn
```

#### GPU Support (NVIDIA only)
```bash
# Ensure CUDA 11.2+ and cuDNN 8.1+ are installed
pip install tensorflow-gpu>=2.8.0
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mtcnn; print('MTCNN: Installed')"
```

Expected output:
```
TensorFlow: 2.x.x
OpenCV: 4.x.x
MTCNN: Installed
```

---

## ⚡ Quick Start

### Option 1: Use Pre-trained Model (Fastest)

If you have the pre-trained model file:

```bash
# Place Emotion_Detection.h5 in project directory
python test_advanced.py
```

**What happens:**
1. Window opens showing webcam feed
2. Faces detected with blue boxes
3. Emotion labels appear with confidence
4. Probability bars show all emotion scores
5. FPS counter displays in top-left
6. Emotions logged to `emotion_log.csv`

**Controls:**
- `q` - Quit application
- `b` - Toggle probability bars on/off
- `l` - Toggle logging on/off
- `s` - Save screenshot

### Option 2: Train Your Own Model

```bash
# 1. Organize your dataset (see Dataset Structure section)
# 2. Update paths in train_advanced.py
# 3. Run training
python train_advanced.py
```

---

## 📖 Usage Guide

### Running Detection

#### Basic Usage
```bash
python test_advanced.py
```

#### With Custom Model
```python
# Edit test_advanced.py, line 21
MODEL_PATH = './models/your_custom_model.h5'
```

### Understanding the Interface

```
┌─────────────────────────────────────────────────────────────┐
│  FPS: 28.3                    Emotion Probability Bars     │
│  Faces: 1                     ┌─────────────────────────┐   │
│  Frame: 542                   │ Happy:    82.3% ████████ │   │
│                               │ Neutral:  10.1% █        │   │
│  ┌──────────────────┐         │ Surprise:  5.2%          │   │
│  │                  │         │ Sad:       1.8%          │   │
│  │   [Happy 82.3%]  │         │ Angry:     0.6%          │   │
│  │                  │         └─────────────────────────┘   │
│  └──────────────────┘                                      │
│                                                             │
│  [B]ars: ON  [L]og: ON                                     │
└─────────────────────────────────────────────────────────────┘
```

### CSV Logging

Emotions are automatically logged to `emotion_log.csv`:

```csv
timestamp,emotion,confidence,all_probabilities
2024-02-18 10:15:32.123,Happy,0.823,0.006,0.823,0.101,0.018,0.052
2024-02-18 10:15:32.156,Happy,0.831,0.005,0.831,0.095,0.015,0.054
```

**Columns:**
- `timestamp`: Detection time (millisecond precision)
- `emotion`: Predicted emotion label
- `confidence`: Confidence score (0-1)
- `all_probabilities`: Comma-separated probabilities for all 5 emotions

---

## 🎓 Training Guide

### Dataset Structure

Organize your dataset as follows:

```
fer2013/
├── train/
│   ├── Angry/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
│
└── validation/
    ├── Angry/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprise/
```

**Image Requirements:**
- Format: JPG, PNG
- Size: Any (will be resized to 224×224 or 48×48)
- Color: RGB or Grayscale
- Naming: Any valid filename

### Training Configuration

Edit `train_advanced.py` to customize training:

```python
class TrainingConfig:
    # Dataset paths
    TRAIN_DIR = './fer2013/train'
    VAL_DIR = './fer2013/validation'
    
    # Model settings
    IMG_SIZE = 224  # Image size (224 or 48)
    MODEL_ARCHITECTURE = 'mobilenet'  # 'mobilenet', 'efficientnet', or 'resnet'
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    INITIAL_LR = 0.001
    
    # Options
    USE_DATA_AUGMENTATION = True
    FREEZE_BASE_LAYERS = False  # Set True to freeze pre-trained layers
```

### Running Training

```bash
python train_advanced.py
```

**Training Process:**

1. **Initialization** (1-2 minutes)
   - Load and verify dataset
   - Build model architecture
   - Setup callbacks

2. **Training** (2-8 hours depending on hardware)
   - Progress bars show epoch completion
   - Metrics displayed after each epoch
   - Best model automatically saved

3. **Completion**
   - Training history plotted
   - Best model identified
   - Final model saved

**Output Files:**
```
models/
├── mobilenet_20240218_101532_best.h5    # Best model (by val_accuracy)
├── mobilenet_20240218_101532_final.h5   # Final model
└── mobilenet_20240218_101532_history.png # Training plots

logs/
├── mobilenet_20240218_101532/            # TensorBoard logs
└── mobilenet_20240218_101532_history.csv # Training metrics
```

### Monitoring Training with TensorBoard

```bash
# In a separate terminal
tensorboard --logdir=./logs

# Open browser to: http://localhost:6006
```

**TensorBoard Features:**
- Real-time training metrics
- Loss and accuracy curves
- Learning rate schedule
- Model graph visualization
- Distribution of weights

### Training Tips

**For Better Accuracy:**
- Use more training data (20,000+ images recommended)
- Increase `EPOCHS` to 100
- Use data augmentation (`USE_DATA_AUGMENTATION = True`)
- Try different architectures

**For Faster Training:**
- Reduce `IMG_SIZE` to 48
- Increase `BATCH_SIZE` (if you have enough GPU memory)
- Use `FREEZE_BASE_LAYERS = True` (train only top layers)
- Enable mixed precision: `USE_MIXED_PRECISION = True`

**For Better Generalization:**
- Use cross-validation
- Collect diverse dataset
- Increase dropout rates
- Use early stopping

---

## ⚙️ Configuration

### Detection Configuration

Edit constants in `test_advanced.py`:

```python
# Configuration
MODEL_PATH = './Emotion_Detection.h5'
CASCADE_PATH = './haarcascade_frontalface_default.xml'
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display (0.0-1.0)
SHOW_PROBABILITY_BARS = True  # Show emotion probability bars
ENABLE_LOGGING = True  # Log detections to CSV
```

### Smoothing Window

Adjust temporal smoothing:

```python
# Larger window = smoother but slower response
smoother = EmotionSmoother(window_size=7)  # Default: 7 frames

# window_size=3  : Fast response, more jitter
# window_size=7  : Balanced (recommended)
# window_size=15 : Very smooth, delayed response
```

### Face Detection Sensitivity

For Haar Cascade:

```python
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.3,  # Lower = more sensitive (1.1-1.5)
    minNeighbors=5,   # Higher = fewer false positives (3-6)
    minSize=(30, 30)  # Minimum face size
)
```

For MTCNN:

```python
if detection['confidence'] > 0.9:  # Higher = fewer false positives (0.7-0.95)
```

---

## 📊 Performance

### Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **FPS** | 20-30 | On modern CPU (Intel i7/AMD Ryzen) |
| **FPS (GPU)** | 50-100 | With NVIDIA GPU (GTX 1060+) |
| **Detection Latency** | 30-50ms | Per frame |
| **Face Detection** | 95%+ | Frontal faces, good lighting |
| **Emotion Accuracy** | 65-75% | On validation set |
| **Model Size** | 15-90MB | Depends on architecture |

### Per-Emotion Accuracy (Typical)

| Emotion | Accuracy | Notes |
|---------|----------|-------|
| **Happy** | 75-85% | Best performance |
| **Surprise** | 70-80% | Good performance |
| **Angry** | 65-75% | Moderate |
| **Sad** | 60-70% | Often confused with Neutral |
| **Neutral** | 55-65% | Most challenging |

### Hardware Recommendations

#### Minimum (Detection Only)
- CPU: Intel i5 or AMD equivalent
- RAM: 8GB
- Webcam: 720p
- Performance: 15-20 FPS

#### Recommended (Detection)
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB
- Webcam: 1080p
- Performance: 25-30 FPS

#### Recommended (Training)
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 16-32GB
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)
- Storage: SSD with 50GB+ free
- Training Time: 2-4 hours

---

## 🔧 Troubleshooting

### Common Issues

#### 1. MTCNN Not Available

**Symptom:** "MTCNN not available, using Haar Cascade"

**Solution:**
```bash
pip install mtcnn
```

**Workaround:** System works fine with Haar Cascade (default fallback)

#### 2. Low FPS / Lag

**Solutions:**

**A. Reduce Resolution**
```python
# In test_advanced.py, after cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**B. Disable Probability Bars**
- Press `b` key during runtime, or
- Set `SHOW_PROBABILITY_BARS = False`

**C. Increase Smoothing Window**
```python
smoother = EmotionSmoother(window_size=3)  # Faster response
```

**D. Use GPU**
```bash
pip install tensorflow-gpu
```

#### 3. No Faces Detected

**Solutions:**

**A. Improve Lighting**
- Ensure face is well-lit
- Avoid backlighting
- Use indirect lighting

**B. Adjust Detection Sensitivity**
```python
# Make detection more sensitive
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,  # More sensitive
    minNeighbors=3,   # More detections
    minSize=(20, 20)  # Smaller faces
)
```

**C. Try MTCNN**
```bash
pip install mtcnn
```

#### 4. "Cannot open webcam"

**Solutions:**

**A. Check Camera Index**
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

**B. Check Permissions**
- **macOS:** System Preferences → Security & Privacy → Camera
- **Linux:** `sudo usermod -a -G video $USER`
- **Windows:** Settings → Privacy → Camera

**C. Close Other Applications**
- Close Zoom, Skype, Teams
- Close other camera apps

#### 5. Out of Memory During Training

**Solutions:**

**A. Reduce Batch Size**
```python
BATCH_SIZE = 16  # or 8
```

**B. Reduce Image Size**
```python
IMG_SIZE = 48  # Instead of 224
```

**C. Enable GPU Memory Growth**
```python
# Add to train_advanced.py
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 6. Predictions Are Unstable

**Solution:** Increase smoothing window
```python
smoother = EmotionSmoother(window_size=15)
```

---

## 📚 API Documentation

### EmotionSmoother Class

```python
class EmotionSmoother:
    """
    Smooths emotion predictions over time using rolling average
    
    Args:
        window_size (int): Number of frames to average (default: 7)
    
    Methods:
        smooth(prediction): Apply smoothing to prediction array
        reset(): Clear prediction history
    """
```

**Example:**
```python
smoother = EmotionSmoother(window_size=7)
smoothed = smoother.smooth(predictions)
```

### EmotionLogger Class

```python
class EmotionLogger:
    """
    Logs emotion detections to CSV file
    
    Args:
        log_file (str): Path to CSV log file (default: 'emotion_log.csv')
    
    Attributes:
        enabled (bool): Enable/disable logging
    
    Methods:
        log(emotion, confidence, probabilities): Log single detection
    """
```

**Example:**
```python
logger = EmotionLogger('my_log.csv')
logger.log('Happy', 0.85, [0.05, 0.85, 0.05, 0.03, 0.02])
logger.enabled = False  # Disable logging
```

### FPSCounter Class

```python
class FPSCounter:
    """
    Calculates frames per second
    
    Args:
        window_size (int): Number of frames for averaging (default: 30)
    
    Methods:
        update(): Call once per frame
        get_fps(): Returns current FPS
    """
```

**Example:**
```python
fps_counter = FPSCounter(window_size=30)
while True:
    fps_counter.update()
    fps = fps_counter.get_fps()
    print(f"FPS: {fps:.1f}")
```

---

## 📈 Comparison: v1.0 vs v2.0

| Feature | v1.0 (Basic) | v2.0 (Advanced) |
|---------|--------------|-----------------|
| **Face Detection** | Haar Cascade only | MTCNN + Haar Cascade |
| **Prediction Stability** | Jittery | Smooth (temporal averaging) |
| **Confidence Display** | No | Yes (with threshold) |
| **Probability Visualization** | No | Yes (optional bars) |
| **Logging** | No | Yes (CSV with timestamps) |
| **Multi-face Support** | Basic | Enhanced with tracking |
| **Performance Monitoring** | No | FPS counter included |
| **Interactive Controls** | Quit only | Toggle features in real-time |
| **Screenshot Capture** | No | Yes ('s' key) |
| **Model Architectures** | MobileNet only | MobileNet/EfficientNet/ResNet |
| **Training Monitoring** | Basic | TensorBoard + Plots |
| **Data Augmentation** | Basic | Enhanced (8 techniques) |
| **Callbacks** | 3 callbacks | 5 callbacks |
| **Model Saving** | Basic | Best + Final models |
| **Batch Normalization** | No | Yes |
| **Dropout Strategy** | Single rate | Graduated rates |
| **Code Organization** | Functional | Object-oriented |
| **Error Handling** | Basic | Comprehensive |
| **Documentation** | Basic | Extensive |

**Performance Improvements:**
- **Accuracy**: +5-10% (with better training)
- **Stability**: 3x more stable predictions
- **FPS**: Similar (optimized pipeline)
- **User Experience**: Significantly improved

---

## 🚀 Future Roadmap

### Planned for v2.1 (Next Release)
- [ ] 7-emotion support (add Fear, Disgust)
- [ ] Face landmark detection
- [ ] Micro-expression detection
- [ ] Gender and age estimation
- [ ] Video file input support

### Planned for v3.0 (Major Update)
- [ ] Transformer-based architecture (ViT)
- [ ] Temporal 3D CNN for video understanding
- [ ] Multi-modal emotion recognition (audio + visual)
- [ ] REST API server
- [ ] Web interface (Flask/FastAPI)
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Real-time analytics dashboard

### Research Features (Experimental)
- [ ] Attention mechanisms
- [ ] Self-supervised pre-training
- [ ] Few-shot learning
- [ ] Domain adaptation
- [ ] Explainable AI (GradCAM heatmaps)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Types of Contributions

1. **Bug Reports**: Found a bug? Open an issue
2. **Feature Requests**: Have an idea? Share it
3. **Code Contributions**: Submit a pull request
4. **Documentation**: Improve README or comments
5. **Testing**: Test on different hardware/datasets

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/emotion-detection-v2.git
cd emotion-detection-v2

# Create branch
git checkout -b feature-name

# Make changes and commit
git add .
git commit -m "Add feature: description"

# Push and create PR
git push origin feature-name
```

### Code Style

- Follow PEP 8
- Add docstrings to functions
- Include type hints where appropriate
- Write descriptive commit messages

---

## 📄 License

MIT License - See LICENSE file for details

```
Copyright (c) 2024 Emotion Detection Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software")...
```

---

## 🙏 Acknowledgments

- **Datasets**: FER2013, AffectNet, RAF-DB communities
- **Pre-trained Models**: TensorFlow/Keras team
- **MTCNN**: Joint Cascade Face Detection and Alignment
- **OpenCV**: Open Source Computer Vision Library
- **Contributors**: All who have contributed to this project

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/emotion-detection-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/emotion-detection-v2/discussions)
- **Documentation**: This README + code comments
- **Email**: support@emotiondetection.com (if applicable)

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{emotion_detection_v2,
  title={Advanced Real-Time Emotion Detection System},
  author={Your Name},
  year={2024},
  version={2.0},
  url={https://github.com/yourusername/emotion-detection-v2}
}
```

---

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for AI and Computer Vision Enthusiasts**

*Version 2.0 - Last Updated: February 2024*

---

## 📚 Additional Resources

### Learn More
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [OpenCV Python Documentation](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Emotion Recognition Papers](https://paperswithcode.com/task/facial-expression-recognition)

### Datasets
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [RAF-DB](http://www.whdeng.cn/raf/model1.html)
- [ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)

### Related Projects
- [DeepFace](https://github.com/serengil/deepface)
- [FaceNet](https://github.com/davidsandberg/facenet)
- [OpenFace](https://cmusatyalab.github.io/openface/)

---