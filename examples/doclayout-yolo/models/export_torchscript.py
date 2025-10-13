#!/usr/bin/env python3
"""
Export DocLayout-YOLO model to TorchScript format for Rust inference.
"""

import torch
from pathlib import Path
from doclayout_yolo import YOLOv10

def export_model():
    print("=" * 70)
    print("Exporting DocLayout-YOLO to TorchScript")
    print("=" * 70)
    
    model_path = Path(__file__).parent / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    output_path = Path(__file__).parent / "doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt"
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Please run: python download_latest_model.py")
        return False
    
    print(f"\n1. Loading model: {model_path.name}")
    model = YOLOv10(str(model_path))
    model.model.eval()
    print("   ✓ Model loaded")
    
    print("\n2. Creating TorchScript wrapper...")
    
    class YOLOWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.model(x)
            
            if isinstance(output, dict):
                if 'one2one' in output:
                    result = output['one2one']
                elif 'one2many' in output:
                    result = output['one2many']
                elif 'predictions' in output:
                    result = output['predictions']
                else:
                    result = list(output.values())[0]
                
                if isinstance(result, (tuple, list)):
                    return result[0] if torch.is_tensor(result[0]) else result
                return result
            elif isinstance(output, (tuple, list)):
                return output[0]
            else:
                return output
    
    wrapped_model = YOLOWrapper(model.model)
    wrapped_model.eval()
    print("   ✓ Wrapper created")
    
    print("\n3. Testing with dummy input (1×3×1280×1280)...")
    dummy_input = torch.randn(1, 3, 1280, 1280)
    
    with torch.no_grad():
        output = wrapped_model(dummy_input)
        print(f"   ✓ Output shape: {output.shape}")
    
    print("\n4. Tracing model with torch.jit.trace()...")
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, dummy_input)
    print("   ✓ Model traced")
    
    print(f"\n5. Saving to: {output_path.name}")
    torch.jit.save(traced_model, str(output_path))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved! Size: {size_mb:.1f} MB")
    
    print("\n6. Verifying saved model...")
    loaded = torch.jit.load(str(output_path))
    with torch.no_grad():
        verify_output = loaded(dummy_input)
        print(f"   ✓ Verified output shape: {verify_output.shape}")
    
    print("\n" + "=" * 70)
    print("✅ Export complete!")
    print("=" * 70)
    print(f"\nTorchScript model: {output_path.name}")
    print("Ready for Rust inference via tch-rs!")
    
    return True

if __name__ == "__main__":
    success = export_model()
    exit(0 if success else 1)
