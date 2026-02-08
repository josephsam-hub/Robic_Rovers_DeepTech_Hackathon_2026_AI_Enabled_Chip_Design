# Wafer Defect Detection

**Edge-AI Semiconductor Wafer/Die Defect Classification System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

---

## 📋 Overview

An industrial-grade deep learning system for real-time semiconductor wafer defect detection and classification. Built for edge deployment with SqueezeNet 1.1, achieving **94-96% accuracy** in a **2.91 MB** model suitable for production line inspection.

### Key Features
- ✅ **Ultra-lightweight**: 2.91 MB model fits entirely in processor SRAM
- ✅ **Real-time inference**: High-throughput edge deployment
- ✅ **Explainable AI**: Grad-CAM visualization for defect localization
- ✅ **Uncertainty-aware**: Confidence scoring and entropy-based rejection
- ✅ **Production-ready**: ONNX export, quantization support

---

## 🎯 Defect Classes

The system classifies 8 types of semiconductor defects plus clean wafers:

| Class | Description | Count |
|-------|-------------|-------|
| **Clean** | Non-defective wafer | 187 |
| **Bridge** | Unwanted connection between circuit lines | 75 |
| **Crack** | Physical cracks on wafer surface | 103 |
| **LER** | Line Edge Roughness defects | 60 |
| **Line Collapse** | Collapsed resist lines | 51 |
| **LWV** | Line Width Variation defects | 56 |
| **Open** | Broken circuit connections | 50 |
| **Scratch** | Surface scratches | 37 |
| **Via** | Via connection defects | 20 |

**Total Dataset**: ~1,517 SEM images (256×256 grayscale)

---

## 🏗️ Architecture: SqueezeNet 1.1

### Why SqueezeNet?

SqueezeNet 1.1 provides the optimal **accuracy-to-footprint ratio** for edge deployment:

```
Input (256×256×1 grayscale)
    ↓
Conv2d (Initial convolution)
    ↓
Fire Modules (8 blocks)
  ├─ Squeeze: 1×1 conv (compression)
  └─ Expand: 1×1 + 3×3 conv (feature extraction)
    ↓
Max Pooling
    ↓
Final Conv2d
    ↓
Global Average Pooling
    ↓
Softmax (9 classes)
```

### Design Principles
1. **Replace 3×3 with 1×1 filters** → 9x fewer parameters
2. **Reduce input channels** → Lower computation
3. **Downsample late** → Preserve spatial information

### Model Comparison

| Model | Size | Parameters | Accuracy | Edge Fit | Cache Fit |
|-------|------|------------|----------|----------|-----------|
| **SqueezeNet 1.1** | **2.91 MB** ✅ | 1.24M | 94-96% | Excellent | Yes ✅ |
| EfficientNet-Lite0 | 14 MB | 4.7M | High | Good | No |
| ResNet-18 | 45 MB | 11M | High | Poor | No |
| MobileNetV2 | 14 MB | 3.5M | High | Good | No |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Wafer-Defect-Detection.git
cd Wafer-Defect-Detection
pip install -r requirements.txt
```

### Training

```bash
python src/train.py --data data/Datasets --epochs 50 --batch-size 32
```

### Inference

```bash
python src/inference.py --model models/squeezenet_final_2_91MB.pth --image path/to/wafer.png
```

### Evaluation

```bash
python src/evaluate.py --model models/squeezenet_final_2_91MB.pth --data data/Datasets
```

---

## 📊 Performance

### Metrics
- **Overall Accuracy**: 94-96% (major classes)
- **Model Size**: 2.91 MB (FP32)
- **Inference Speed**: Real-time capable
- **Memory Footprint**: Fits in SRAM (<5MB)

### Evaluation Features
- Confusion matrix visualization
- Per-class accuracy reports
- Top-3 prediction confidence
- Entropy-based uncertainty detection
- Grad-CAM heatmap generation

---

## 🔬 Experiments

Detailed results and comparisons available in [`experiments/`](experiments/):

- [SqueezeNet Results](experiments/squeezenet_results.md) - Final model architecture and performance
- [EfficientNet Results](experiments/efficientnet_results.md) - Comparison analysis
- [ResNet Results](experiments/resnet_results.md) - Baseline comparison
- [Challenges Faced](experiments/challenges_faced.md) - Dataset limitations and solutions

---

## 📦 Deployment

### ONNX Export

```bash
python deployment/export_to_onnx.py --model models/squeezenet_final_2_91MB.pth
```

### Edge Inference Demo

```bash
python deployment/edge_inference_demo.py --onnx models/onnx_model.onnx
```

### Supported Platforms
- ✅ TensorRT (NVIDIA)
- ✅ CoreML (Apple)
- ✅ TFLite (Google)
- ✅ OpenVINO (Intel)
- ✅ NXP eIQ (Target platform)

---

## 📂 Project Structure

```
Wafer-Defect-Detection/
├── data/
│   ├── Datasets/              # Wafer defect images
│   └── dataset_description.md # Dataset documentation
├── src/
│   ├── dataset.py            # Data loading and augmentation
│   ├── model.py              # SqueezeNet architecture
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── inference.py          # Inference with Grad-CAM
│   └── gradcam.py            # Explainability module
├── experiments/
│   ├── squeezenet_results/   # Training results
│   ├── efficientnet_results/ # Comparison results
│   └── *.md                  # Experiment documentation
├── models/
│   ├── squeezenet_final_2_91MB.pth  # Final model
│   └── onnx_model.onnx              # ONNX export
├── deployment/
│   ├── export_to_onnx.py     # Model conversion
│   └── edge_inference_demo.py # Edge deployment demo
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_training_squeezenet.ipynb
│   └── 03_model_comparison.ipynb
└── requirements.txt
```

---

## 🎓 Training Best Practices

### Data Augmentation
- Random rotation (±15°)
- Horizontal/vertical flips
- Brightness/contrast adjustment
- Gaussian noise injection

### Training Strategy
- **Optimizer**: AdamW (weight decay 1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss (class-weighted)
- **Split**: 70% train / 15% val / 15% test (stratified)
- **Transfer Learning**: ImageNet pretrained weights

### Regularization
- Dropout (0.5)
- Early stopping (patience=10)
- Data augmentation
- Weight decay

---

## 🔍 Explainability

### Grad-CAM Visualization
The system includes Grad-CAM (Gradient-weighted Class Activation Mapping) for visual explanation:

```python
from src.gradcam import generate_gradcam
heatmap = generate_gradcam(model, image, target_class)
```

### Uncertainty Quantification
- **Confidence Threshold**: Reject predictions below 70%
- **Entropy-based Detection**: Flag high-uncertainty samples
- **Top-3 Predictions**: Display alternative classifications

---

## 🚧 Challenges & Solutions

### Dataset Limitations
- **Challenge**: Limited public wafer defect datasets
- **Solution**: Obtained images from IEEE papers, explored synthetic generation

### Class Imbalance
- **Challenge**: Uneven distribution (187 clean vs 20 via defects)
- **Solution**: Stratified splitting, weighted loss, data augmentation

### Edge Constraints
- **Challenge**: <5MB model size requirement
- **Solution**: SqueezeNet 1.1 with Fire Modules (2.91 MB)

See [challenges_faced.md](experiments/challenges_faced.md) for complete details.

---

## 🔮 Future Improvements

- [ ] Multi-scale defect detection
- [ ] Instance segmentation for defect localization
- [ ] Active learning for data-efficient training
- [ ] Quantization to INT8 (further size reduction)
- [ ] Real-time video stream processing
- [ ] Integration with fab inspection systems

---

## 📝 Phase-2 Enhancement Note

> **Note**: Additional industrial wafer defect datasets will be received after the hackathon deadline. If selected for Phase-2, the model will be retrained and fine-tuned on expanded data to improve detection accuracy and robustness across diverse manufacturing conditions.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IESA DeepTech Hackathon** - Challenge framework
- **NXP Semiconductors** - Target edge platform (eIQ)
- **IEEE DataPort** - Dataset sources
- **PyTorch Community** - Deep learning framework

---

## 📧 Contact

For questions or collaboration:
- **Project**: Robic Rovers Team
- **Event**: IESA DeepTech Hackathon 2026
- **Focus**: AI-Enabled Chip Design

---

**Built with ❤️ for semiconductor manufacturing quality control**
