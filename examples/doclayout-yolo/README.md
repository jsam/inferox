# DocLayout-YOLO Example

This example demonstrates how to use **TorchScript models** with Inferox using the DocLayout-YOLO document layout detection model.

## Overview

DocLayout-YOLO is a state-of-the-art document layout detection model that identifies and localizes different elements in document images (text blocks, figures, tables, etc.). This example shows how to:

1. Load TorchScript (`.pt`) models with Inferox
2. Run inference with the Tch backend
3. Integrate TorchScript models with the Inferox Engine API

## Key Features

- ✅ **Zero architecture code required** - Just load the `.pt` file!
- ✅ **Single-line model loading** - `DocLayoutYOLO::from_pretrained(path, device)`
- ✅ **Engine integration** - Works seamlessly with InferoxEngine
- ✅ **Batch inference support** - Process multiple images efficiently
- ✅ **Device flexibility** - CPU, CUDA, or MPS support

## Model Details

- **Name**: DocLayout-YOLO
- **Source**: [HuggingFace - juliozhao/DocLayout-YOLO-DocStructBench](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench)
- **Format**: TorchScript (`.pt`)
- **Size**: 38.8 MB
- **Input**: RGB images (1024×1024)
- **License**: Apache-2.0

## Quick Start

### 1. Download and Export the Model

First, download the original PyTorch checkpoint and export it to TorchScript:

```bash
cd examples/doclayout-yolo/models

# Download the original model (38.8 MB)
curl -L -o doclayout_yolo_docstructbench_imgsz1024.pt \
  "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt"

# Install the doclayout-yolo package
pip install doclayout-yolo

# Export to TorchScript format (78.1 MB)
python export_to_torchscript.py
# => Creates doclayout_yolo_torchscript.pt

ls -lh
# => doclayout_yolo_torchscript.pt (78.1 MB - ready for Rust!)
```

**Note**: The original `.pt` file from HuggingFace is a PyTorch checkpoint that requires Python. The export script converts it to a standard TorchScript file that can be loaded directly in Rust with zero Python dependencies.

### 2. Run Tests

```bash
# Set environment variables for LibTorch
export LIBTORCH_USE_PYTORCH=1
export DYLD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$DYLD_LIBRARY_PATH

# Run all E2E tests
cargo test --package doclayout-yolo --test e2e_test -- --ignored --nocapture

# Run specific tests
cargo test --package doclayout-yolo --test e2e_test test_doclayout_yolo_direct_inference -- --ignored --nocapture
cargo test --package doclayout-yolo --test e2e_test test_doclayout_yolo_with_engine -- --ignored --nocapture
```

## Usage Examples

### Direct Inference

```rust
use doclayout_yolo::DocLayoutYOLO;
use inferox_core::Model;
use tch::Device;

// Load model
let model = DocLayoutYOLO::from_pretrained(
    "models/doclayout_yolo_docstructbench_imgsz1024.pt",
    Device::Cpu
)?;

// Run inference
let detections = model.forward(image_tensor)?;

println!("Detections: {:?}", detections.shape());
```

### Engine Integration

```rust
use inferox_engine::{InferoxEngine, EngineConfig};
use doclayout_yolo::DocLayoutYOLO;
use tch::Device;

// Create engine
let mut engine = InferoxEngine::new(EngineConfig::default());

// Load and register model
let model = DocLayoutYOLO::from_pretrained("model.pt", Device::Cpu)?;
engine.register_model("doc-detector", Box::new(model), None);

// Backend-agnostic inference
let output = engine.infer_typed::<TchBackend>("doc-detector", image_tensor)?;
```

### Batch Inference

```rust
let model = DocLayoutYOLO::from_pretrained("model.pt", Device::Cpu)?;

// Process multiple images at once
let images = vec![image1, image2, image3];
let results = model.detect_batch(images)?;

for (i, result) in results.iter().enumerate() {
    println!("Image {}: detected {} objects", i, result.shape()[0]);
}
```

## API Reference

### `DocLayoutYOLO`

```rust
impl DocLayoutYOLO {
    /// Load model from TorchScript file
    pub fn from_pretrained(
        model_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, tch::TchError>;
    
    /// Detect document layout elements in a single image
    pub fn detect(&self, image: TchTensor) -> Result<TchTensor, tch::TchError>;
    
    /// Detect document layout elements in multiple images
    pub fn detect_batch(&self, images: Vec<TchTensor>) 
        -> Result<Vec<TchTensor>, tch::TchError>;
}
```

### `Model` Trait Implementation

```rust
impl Model for DocLayoutYOLO {
    type Backend = TchBackend;
    type Input = TchTensor;
    type Output = TchTensor;
    
    fn name(&self) -> &str;
    fn metadata(&self) -> ModelMetadata;
    fn forward(&self, input: Self::Input) -> Result<Self::Output, tch::TchError>;
}
```

## Architecture Comparison

### Before: SafeTensors + Manual Implementation

With SafeTensors, you'd need to implement the entire YOLO architecture in Rust:

```rust
// 500+ lines of YOLO architecture code
pub struct YOLOModel {
    backbone: Backbone,          // CSPDarknet implementation
    neck: Neck,                  // PANet implementation  
    head: DetectionHead,         // Detection head implementation
    // ... plus all the layer implementations
}

impl YOLOModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // 1. Backbone (100+ lines)
        // 2. Neck (100+ lines)
        // 3. Head (100+ lines)
        // 4. NMS (50+ lines)
        // ...
    }
}
```

### After: TorchScript

With TorchScript, it's just 10 lines:

```rust
use inferox_tch::TorchScriptModel;

let model = TorchScriptModel::load("yolo.pt", Device::Cpu)?;
let output = model.forward(input)?;
```

**That's a 98% code reduction!** 🎉

## TorchScript Benefits

1. **No Architecture Code**: Model architecture is embedded in the `.pt` file
2. **Fast Loading**: Pre-compiled and optimized graphs
3. **Version Independence**: Works across PyTorch versions
4. **Custom Ops Support**: Can include custom CUDA kernels
5. **Production Ready**: Used by major companies (Meta, Uber, etc.)

## Input Format

DocLayout-YOLO expects:
- **Shape**: `[batch_size, 3, 1024, 1024]`
- **Type**: Float32
- **Range**: `[0.0, 1.0]` (normalized RGB)
- **Format**: RGB (not BGR)

Example preprocessing:

```rust
use tch::{Tensor, Kind, Device};

// Load image (assume image is H×W×3 RGB)
let image = tch::vision::image::load("document.jpg")?;

// Resize to 1024×1024
let resized = image.resize2d(1024, 1024);

// Normalize to [0, 1]
let normalized = resized.to_kind(Kind::Float) / 255.0;

// Add batch dimension
let batched = normalized.unsqueeze(0);
```

## Output Format

The model outputs detection results with:
- Bounding boxes (x1, y1, x2, y2)
- Class labels
- Confidence scores

Output shape: `[batch_size, num_detections, 6]` where each detection is `[x1, y1, x2, y2, confidence, class]`

## Performance

On MacBook Pro M1:
- **Load Time**: ~100ms (first time), ~10ms (cached)
- **Inference Time**: ~200ms per image (CPU)
- **Memory Usage**: ~150MB

With CUDA (if available):
- **Inference Time**: ~20-50ms per image

## Troubleshooting

### Model Not Found

```
Model not found at "models/doclayout_yolo_docstructbench_imgsz1024.pt"
```

**Solution**: The model should be automatically downloaded. If not:
```bash
cd examples/doclayout-yolo/models
curl -L -o doclayout_yolo_docstructbench_imgsz1024.pt \
  "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt"
```

### LibTorch Not Found

```
error: failed to run custom build command for `torch-sys`
```

**Solution**: Install PyTorch with LibTorch:
```bash
# macOS
pip install torch torchvision

# Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA Errors

```
CUDA is not available
```

**Solution**: Either:
1. Use CPU: `Device::Cpu`
2. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Extending This Example

### Use Your Own TorchScript Model

```rust
// Export your model in Python
// model = YourModel()
// scripted = torch.jit.script(model)
// torch.jit.save(scripted, "your_model.pt")

// Load in Rust
let model = TorchScriptModel::load("your_model.pt", Device::Cpu)?;
let output = model.forward(input)?;
```

### Add Preprocessing/Postprocessing

```rust
pub struct DocLayoutYOLOWithPreprocessing {
    model: DocLayoutYOLO,
}

impl DocLayoutYOLOWithPreprocessing {
    pub fn detect_from_image(&self, image_path: &str) -> Result<Vec<Detection>> {
        // 1. Load and preprocess
        let tensor = self.preprocess(image_path)?;
        
        // 2. Run model
        let output = self.model.forward(tensor)?;
        
        // 3. Postprocess (NMS, thresholding, etc.)
        let detections = self.postprocess(output)?;
        
        Ok(detections)
    }
}
```

## Related Examples

- **bert-tch**: SafeTensors loading example
- **bert-candle**: Candle backend example
- **mlp**: Simple model example

## References

- [DocLayout-YOLO Paper](https://arxiv.org/abs/2410.12628)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [tch-rs Documentation](https://docs.rs/tch)
- [Inferox Documentation](../../README.md)

## License

This example code is MIT licensed. The DocLayout-YOLO model is Apache-2.0 licensed.
