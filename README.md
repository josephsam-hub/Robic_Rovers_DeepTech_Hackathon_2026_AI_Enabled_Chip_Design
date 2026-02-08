# 🏆 IESA DeepTech Hackathon 2026 - AI-Enabled Chip Design

**Team**: Robic Rovers  
**Challenge**: Semiconductor Wafer/Die Defect Classification using Edge-AI

---

## 🎯 Project Overview

This repository contains our solution for the **IESA DeepTech Hackathon 2026**, focused on building an **Edge-AI defect classification system** for semiconductor manufacturing quality control.

### Challenge Requirements
- ✅ Classify wafer/die SEM images into **8+ defect classes**
- ✅ Balance **accuracy**, **model size**, and **edge deployment readiness**
- ✅ Target platform: **NXP eIQ** edge inference
- ✅ Real-time inspection under limited compute resources

---

## 🚀 Our Solution

### Wafer Defect Detection System

We developed an **ultra-lightweight CNN-based defect detection system** using **SqueezeNet 1.1** that achieves:

- **94-96% accuracy** on major defect classes
- **2.91 MB model size** (fits entirely in SRAM)
- **Real-time inference** capability
- **Explainable AI** with Grad-CAM visualization
- **Edge-ready deployment** (ONNX, quantization support)

### Key Innovation
- **Grayscale SEM processing** optimized for texture and structural learning
- **Fire Module architecture** for extreme compression without accuracy loss
- **Uncertainty-aware inference** with confidence scoring
- **Production-ready pipeline** from training to edge deployment

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Model Architecture** | SqueezeNet 1.1 |
| **Model Size** | 2.91 MB |
| **Parameters** | 1.24 Million |
| **Accuracy** | 94-96% |
| **Input Resolution** | 256×256 grayscale |
| **Inference Speed** | Real-time |
| **Edge Compatibility** | Excellent (SRAM-fit) |

### Defect Classes (9 total)
- Clean, Bridge, Crack, LER, Line Collapse, LWV, Open, Scratch, Via

---

## 🏗️ Repository Structure

```
Robic_Rovers_DeepTech_Hackathon_2026_AI_Enabled_Chip_Design/
├── Wafer-Defect-Detection/          # Main project directory
│   ├── data/                        # Dataset and descriptions
│   ├── src/                         # Source code (train, eval, inference)
│   ├── models/                      # Trained models (PyTorch, ONNX)
│   ├── experiments/                 # Model comparisons and results
│   ├── deployment/                  # Edge deployment scripts
│   ├── notebooks/                   # Jupyter notebooks
│   └── README.md                    # Detailed project documentation
└── README.md                        # This file (hackathon overview)
```

---

## 🎓 Technical Highlights

### 1. Model Selection
We evaluated multiple architectures:

| Model | Size | Accuracy | Edge Fit | Selected |
|-------|------|----------|----------|----------|
| **SqueezeNet 1.1** | 2.91 MB | 94-96% | Excellent ✅ | ✅ |
| EfficientNet-Lite0 | 14 MB | High | Good | ❌ |
| ResNet-18 | 45 MB | High | Poor | ❌ |

**Decision**: SqueezeNet 1.1 provides optimal accuracy-to-footprint ratio for edge deployment.

### 2. Architecture Innovation
- **Fire Modules**: Squeeze (1×1) + Expand (1×1 + 3×3) for parameter efficiency
- **Global Average Pooling**: No heavy dense layers, scalable to any resolution
- **Late Downsampling**: Preserves spatial information for defect detection

### 3. Training Pipeline
- Transfer learning from ImageNet
- Stratified train/val/test split
- Data augmentation (rotation, flip, brightness, contrast)
- AdamW optimizer + CosineAnnealing scheduler
- Class-weighted CrossEntropy loss

### 4. Explainability & Uncertainty
- **Grad-CAM**: Visual explanation of defect localization
- **Confidence Scoring**: Top-3 predictions with probabilities
- **Entropy-based Rejection**: Flag uncertain predictions

---

## 🚧 Challenges Overcome

### Dataset Limitations
- **Problem**: Limited public wafer defect datasets (industrial confidentiality)
- **Solution**: Obtained images from IEEE papers, explored synthetic generation

### Class Imbalance
- **Problem**: Uneven distribution (187 clean vs 20 via defects)
- **Solution**: Stratified splitting, weighted loss, augmentation

### Edge Constraints
- **Problem**: <5MB model size requirement for SRAM-fit
- **Solution**: SqueezeNet 1.1 with Fire Modules (2.91 MB)

### Overfitting
- **Problem**: Larger models (ResNet, EfficientNet) overfitted on limited data
- **Solution**: Lightweight architecture with better generalization

---

## 🔮 Phase-2 Enhancement Plan

> **Note**: If selected for Phase-2, we will receive additional industrial wafer defect datasets. Our enhancement plan includes:

1. **Model Retraining**: Fine-tune on expanded dataset for improved robustness
2. **Quantization**: INT8 quantization for further size reduction
3. **Multi-scale Detection**: Handle varying defect sizes
4. **Real-time Video**: Process continuous inspection streams
5. **Fab Integration**: Deploy on actual production line hardware

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/yourusername/Robic_Rovers_DeepTech_Hackathon_2026_AI_Enabled_Chip_Design.git
cd Robic_Rovers_DeepTech_Hackathon_2026_AI_Enabled_Chip_Design/Wafer-Defect-Detection
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

### ONNX Export
```bash
python deployment/export_to_onnx.py --model models/squeezenet_final_2_91MB.pth
```

---

## 📈 Competitive Advantages

1. **Ultra-Lightweight**: 2.91 MB model fits entirely in processor cache
2. **Real-time Performance**: High-throughput inspection without cloud dependency
3. **Industry-Ready**: ONNX export, quantization support, multiple platform compatibility
4. **Cost-Effective**: Reduced inspection cost, no expensive infrastructure
5. **Explainable**: Grad-CAM visualization for quality assurance
6. **Scalable**: Modular pipeline ready for production deployment

---

## 📚 Documentation

Detailed documentation available in [`Wafer-Defect-Detection/`](Wafer-Defect-Detection/):

- [Main Project README](Wafer-Defect-Detection/README.md) - Complete technical documentation
- [Dataset Description](Wafer-Defect-Detection/data/dataset_description.md) - Dataset statistics and structure
- [SqueezeNet Results](Wafer-Defect-Detection/experiments/squeezenet_results.md) - Architecture and performance
- [Model Comparisons](Wafer-Defect-Detection/experiments/) - EfficientNet, ResNet analysis
- [Challenges Faced](Wafer-Defect-Detection/experiments/challenges_faced.md) - Problems and solutions

---

## 🎯 Hackathon Alignment

### IESA DeepTech Challenge Goals
✅ **AI-Enabled Chip Design**: Automated defect detection for semiconductor manufacturing  
✅ **Edge Deployment**: Ultra-lightweight model for on-device inference  
✅ **Real-world Impact**: Production-ready system for fab quality control  
✅ **Innovation**: Fire Module architecture + explainable AI  
✅ **Scalability**: Modular pipeline ready for industrial deployment  

---

## 🏅 What Makes This Solution Stand Out

### Engineering Maturity
- Not just a CNN classifier, but a complete **industrial inspection system**
- Structured experimentation with comparative analysis
- Professional documentation and code organization

### Deployment Mindset
- Model size constraints addressed from day one
- ONNX export and quantization readiness
- Multi-platform compatibility (TensorRT, CoreML, TFLite, OpenVINO)

### Practical Innovation
- Explainability (Grad-CAM) for quality assurance
- Uncertainty modeling for production reliability
- Edge-first design philosophy

---

## 👥 Team: Robic Rovers

**Event**: IESA DeepTech Hackathon 2026  
**Challenge**: AI-Enabled Chip Design  
**Focus**: Semiconductor Wafer Defect Detection  
**Target Platform**: NXP eIQ Edge Inference  

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **IESA DeepTech Hackathon** - Challenge framework and opportunity
- **NXP Semiconductors** - Target edge platform inspiration
- **IEEE DataPort** - Dataset sources and research papers
- **PyTorch Community** - Deep learning framework

---

**Built with ❤️ for the future of semiconductor manufacturing**

*For detailed technical documentation, see [Wafer-Defect-Detection/README.md](Wafer-Defect-Detection/README.md)*
