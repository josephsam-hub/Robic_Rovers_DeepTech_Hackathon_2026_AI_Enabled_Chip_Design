# Challenges Faced

## 1. Dataset Limitations
- **Limited Public Availability**: Industrial wafer defect datasets are confidential and rarely publicly available
- **Access Attempts**: Explored Zenodo, Figshare, HuggingFace, IEEE DataPort, and SCL Semiconductor Laboratory
- **Solution**: Obtained limited defect-class images from IEEE research papers and existing sources

## 2. Class Imbalance
- Uneven distribution across defect categories (e.g., 187 clean vs 20 via defects)
- **Impact**: Risk of model bias toward majority classes
- **Mitigation**: Stratified splitting, data augmentation, weighted loss functions

## 3. Synthetic Data Generation
- **Challenge**: Creating realistic SEM-style defect images for augmentation
- **Issues Faced**:
  - HuggingFace authentication errors
  - Incorrect diffusion pipeline usage
  - Stable Diffusion model loading errors
- **Solution**: Implemented physics-aware procedural generation for bridge, open, and collapse defects

## 4. Model Size vs Accuracy Trade-off
- **Constraint**: Edge deployment requires <5MB model size
- **Challenge**: Balancing accuracy with extreme compression
- **Solution**: Selected SqueezeNet 1.1 (2.91 MB) achieving 94-96% accuracy on major classes

## 5. Overfitting in Larger Models
- ResNet and EfficientNet showed overfitting on limited dataset
- **Solution**: Switched to lightweight SqueezeNet with better generalization

## 6. Explainability Implementation
- **Challenge**: Grad-CAM layer targeting issues with EfficientNet architecture
- **Solution**: Corrected layer selection and implemented proper Grad-CAM visualization

## 7. Edge Deployment Constraints
- Real-time inference requirement
- Limited SRAM on edge devices
- **Solution**: SqueezeNet 1.1 fits entirely in processor cache, enabling high-throughput inspection

## 8. Dataset Noise and Quality
- Varying image quality across sources
- **Impact**: Data quality affected performance more than architecture depth
- **Mitigation**: Careful preprocessing and normalization
