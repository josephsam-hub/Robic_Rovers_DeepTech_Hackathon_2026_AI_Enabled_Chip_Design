# EfficientNet Results

## Model Overview
- **Architecture**: EfficientNet-Lite0
- **Model Size**: ~14-18 MB
- **Parameters**: ~4.7 Million
- **Input Size**: 224×224

## Performance
- **Accuracy**: High (competitive with larger models)
- **Inference Speed**: Medium-High
- **Edge Compatibility**: Good (requires external RAM)

## Key Features
- Compound scaling (depth, width, resolution)
- Inverted residual blocks
- Squeeze-and-excitation modules
- Strong feature extraction

## Challenges Encountered
- **Model Size**: Exceeded 5MB constraint for strict edge deployment
- **Overfitting**: Showed overfitting tendencies on limited dataset
- **Grad-CAM Issues**: Layer targeting complexity for explainability
- **Memory**: Requires external RAM access (cache bottleneck)

## Comparison with SqueezeNet
| Feature | EfficientNet-Lite0 | SqueezeNet 1.1 |
|---------|-------------------|----------------|
| Size | ~14 MB | 2.91 MB ✅ |
| Accuracy | High | 94-96% |
| Edge Fit | Good | Excellent ✅ |
| Cache Fit | No | Yes ✅ |
| Speed | Medium | High ✅ |

## Conclusion
While EfficientNet showed strong performance, SqueezeNet 1.1 was selected for:
- Superior size-to-accuracy ratio
- Better edge deployment characteristics
- Fits entirely in processor SRAM
- Faster inference on resource-constrained devices
