# SqueezeNet 1.1 Results

## Model Overview
- **Architecture**: SqueezeNet 1.1
- **Final Model Size**: 2.91 MB
- **Parameters**: ~1.24 Million
- **Input Size**: 256×256 grayscale

## Performance Metrics
- **Overall Accuracy**: 94-96% (on major defect classes)
- **Inference Speed**: Real-time capable
- **Edge Compatibility**: Excellent (fits in SRAM)

## Why SqueezeNet 1.1?

### Key Advantages
1. **Fire Modules**: Squeeze (1×1) + Expand (1×1 + 3×3) architecture
2. **Model Size**: <5MB - fits entirely in processor cache
3. **Computation**: 2.4x less FLOPs than SqueezeNet 1.0
4. **Global Average Pooling**: No heavy dense layers, scalable to any image size
5. **Cache Efficiency**: Entire model stays in fastest memory during inference

### Design Strategies
- **Strategy 1**: Replace 3×3 filters with 1×1 filters (9x fewer parameters)
- **Strategy 2**: Decrease input channels to 3×3 filters
- **Strategy 3**: Downsample late in network (maintains large feature maps)

## Architecture Flow
```
Input (256×256×1) 
  ↓
Conv2d (Initial)
  ↓
Fire Modules (8 blocks)
  ├─ Squeeze: 1×1 conv
  └─ Expand: 1×1 + 3×3 conv
  ↓
Max Pooling
  ↓
Final Conv2d
  ↓
Global Average Pooling
  ↓
Softmax (8 defect classes)
```

## Training Configuration
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealing
- **Loss**: CrossEntropy
- **Data Split**: Stratified train/val/test
- **Augmentation**: Rotation, flip, brightness, contrast

## Deployment Readiness
- ✅ ONNX export compatible
- ✅ Quantization ready
- ✅ TensorRT, CoreML, TFLite, OpenVINO support
- ✅ Low-latency inference
- ✅ Suitable for NXP eIQ edge deployment

## Competitive Advantage
- **Low Compute**: Minimal memory and computation requirements
- **Real-time**: High-throughput inspection on production line
- **Industry-ready**: No cloud infrastructure needed
- **Cost-effective**: Reduced inspection cost
