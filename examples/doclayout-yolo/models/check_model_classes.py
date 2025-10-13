#!/usr/bin/env python3
"""
Check the actual class names from the DocLayout-YOLO model.
"""

import torch
from pathlib import Path

def check_model():
    print("=" * 70)
    print("Checking DocLayout-YOLO Model Classes")
    print("=" * 70)
    
    # Try to load the PyTorch model
    model_path = Path(__file__).parent / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
    
    print(f"\n1. Loading model: {model_path.name}")
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    
    print("\n2. Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"   - {key}")
    
    if 'model' in checkpoint:
        model = checkpoint['model']
        print(f"\n3. Model type: {type(model)}")
        
        # Try to find class names
        if hasattr(model, 'names'):
            print(f"\n4. Model.names: {model.names}")
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            print(f"\n4. Model.module.names: {model.module.names}")
        else:
            print("\n4. Searching for 'names' in model state...")
            for attr in dir(model):
                if 'name' in attr.lower() and not attr.startswith('_'):
                    try:
                        value = getattr(model, attr)
                        if isinstance(value, (list, dict, tuple)):
                            print(f"   {attr}: {value}")
                    except:
                        pass
    
    # Also check if names are in checkpoint metadata
    if 'names' in checkpoint:
        print(f"\n5. Checkpoint 'names': {checkpoint['names']}")
    
    # Try loading with doclayout_yolo
    print("\n6. Loading with doclayout_yolo package...")
    try:
        from doclayout_yolo import YOLOv10
        model = YOLOv10(str(model_path))
        
        if hasattr(model, 'names'):
            print(f"   YOLOv10.names: {model.names}")
        if hasattr(model.model, 'names'):
            print(f"   YOLOv10.model.names: {model.model.names}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_model()
