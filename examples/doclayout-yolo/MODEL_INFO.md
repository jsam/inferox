# DocLayout-YOLO Model Information

## Current Model

**Model**: `doclayout_yolo_docstructbench_imgsz1280_2501.pt`
- **Source**: HuggingFace - `juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501`
- **Version**: January 2025 (2501)
- **Input Size**: 1280×1280
- **Output Format**: `[1, 14, 33600]`
  - 14 features: `[x_center, y_center, width, height, class0...class9]`
  - 33,600 anchor boxes (vs 21,504 in the old 1024×1024 model)

## Model Classes

The model detects 10 document layout elements:

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | **title** | Document titles, section headers |
| 1 | **plain text** | Body text paragraphs, dense text blocks |
| 2 | **abandon** | Edge elements, page numbers, marginal content |
| 3 | **figure** | Images, charts, diagrams, visual elements |
| 4 | **figure_caption** | Figure captions and labels |
| 5 | **table** | Tables and tabular data |
| 6 | **table_caption** | Table captions and labels |
| 7 | **table_footnote** | Footnotes within tables |
| 8 | **isolate_formula** | Mathematical formulas and equations |
| 9 | **formula_caption** | Formula captions and labels |

## Training Data

Trained on **DocStructBench dataset** which focuses on:
- Scientific papers
- Academic documents
- Research publications

The model is optimized for document layout analysis with clear separation between text, figures, tables, and formulas.

## Setup Instructions

### 1. Download Model (if not present)

```bash
cd models
python download_latest_model.py
```

This downloads the PyTorch model from HuggingFace.

### 2. Export to TorchScript

```bash
python export_torchscript.py
```

This creates `doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt` for Rust inference.

### 3. Run Tests

```bash
# From repository root
make test-doclayout-yolo
```

## Model Files

### PyTorch Model (38 MB)
- **File**: `doclayout_yolo_docstructbench_imgsz1280_2501.pt`
- **Format**: PyTorch checkpoint
- **Use**: Python inference, model export

### TorchScript Model (77 MB)
- **File**: `doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt`
- **Format**: TorchScript (JIT compiled)
- **Use**: Rust inference via tch-rs

## Improvements Over Previous Version

Compared to the old `imgsz1024` model:

1. **Higher Resolution**: 1280×1280 vs 1024×1024
2. **More Anchors**: 33,600 vs 21,504 detection boxes
3. **Better Accuracy**: Improved classification, especially for:
   - List items (charts, figures)
   - Captions and section headers
   - Formula detection
4. **Latest Training**: January 2025 release with updated training

## Performance

On MacBook Pro M1 (CPU):
- **Preprocessing**: ~50ms per image
- **Inference**: ~3-4s per image
- **Postprocessing**: ~10ms per image
- **Total**: ~4s per page (with visualization)

## Postprocessing

The inference pipeline includes:

1. **Non-Maximum Suppression (NMS)**: IOU threshold 0.5
2. **Class-Specific Thresholds**:
   - Footnote: 0.6 (higher to reduce false positives)
   - Text: 0.4 (lower for common elements)
   - Caption/Formula/Picture/Table: 0.6
   - Others: 0.5
3. **Geometric Filtering**:
   - Minimum size: 10×10 pixels
   - Maximum size: 95% of image area
   - Aspect ratio limits: <50:1

## Architecture

- **Base**: YOLOv10 (NMS-free architecture)
- **Framework**: PyTorch + doclayout-yolo package
- **Inference**: TorchScript via tch-rs (LibTorch bindings)
- **Backend**: Inferox TchBackend

## References

- **Paper**: [DocLayout-YOLO: Enhancing Document Layout Analysis](https://arxiv.org/abs/2410.12628)
- **Official Demo**: https://huggingface.co/spaces/opendatalab/DocLayout-YOLO
- **Model Hub**: https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501
- **GitHub**: https://github.com/opendatalab/DocLayout-YOLO
