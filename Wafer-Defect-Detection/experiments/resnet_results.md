# ResNet Results

## Model Overview
- **Architecture**: ResNet-18/34
- **Model Size**: ~45-85 MB
- **Parameters**: ~11-21 Million
- **Input Size**: 224×224

## Performance
- **Accuracy**: High (strong baseline)
- **Inference Speed**: Medium
- **Edge Compatibility**: Poor (too large)

## Key Features
- Residual connections (skip connections)
- Deep architecture (18-34 layers)
- Strong feature learning
- Proven ImageNet performance

## Challenges Encountered
- **Model Size**: Far exceeds edge deployment constraints (>40MB)
- **Overfitting**: Significant overfitting on limited wafer dataset
- **Computation**: High FLOPs requirement
- **Memory**: Cannot fit in edge device SRAM
- **Inference Time**: Slower than lightweight alternatives

## Edge-AI Model Comparison

| Feature | ResNet-18 | SqueezeNet 1.1 | MobileNetV2 | ShuffleNet V2 |
|---------|-----------|----------------|-------------|---------------|
| **Size (FP32)** | ~45 MB | 4.8 MB ✅ | ~14 MB | ~9.2 MB |
| **Parameters** | ~11M | 1.24M ✅ | 3.5M | 2.3M |
| **Key Innovation** | Skip Connections | Fire Modules | Inverted Residuals | Channel Shuffle |
| **Edge Fit** | Poor | Excellent ✅ | Good | Excellent |
| **Cache Fit** | No | Yes ✅ | No | Partial |
| **Inference Speed** | Medium | High ✅ | Medium-High | Very High |
| **Ideal For** | Cloud/Server | Strict Memory <5MB | Mobile Apps | Low-Power ARM |

## Conclusion
ResNet served as a strong accuracy baseline but was rejected for deployment due to:
- Excessive model size (15x larger than SqueezeNet)
- Overfitting on limited dataset
- Incompatible with edge device memory constraints
- Higher computational overhead

**Decision**: SqueezeNet 1.1 provides the optimal accuracy-to-footprint ratio for real-time semiconductor wafer inspection on edge devices.
