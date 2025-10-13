#!/usr/bin/env python3
"""
Download the latest DocLayout-YOLO model from HuggingFace.
This is the same model used in the official demo.
"""

from huggingface_hub import snapshot_download
from pathlib import Path

def download_model():
    print("=" * 70)
    print("Downloading Latest DocLayout-YOLO Model")
    print("=" * 70)
    
    model_repo = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    local_dir = Path(__file__).parent / "DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    
    print(f"\nRepository: {model_repo}")
    print(f"Local directory: {local_dir}")
    print("\nDownloading... (this may take a few minutes)")
    
    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        
        print("\n✅ Download complete!")
        
        model_file = local_dir / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"\nModel file: {model_file.name}")
            print(f"Size: {size_mb:.1f} MB")
            
            target = Path(__file__).parent / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
            if not target.exists():
                import shutil
                shutil.copy(model_file, target)
                print(f"\n✅ Copied to: {target.name}")
        else:
            print(f"\n⚠️  Expected model file not found: {model_file}")
            print("Available files:")
            for f in local_dir.iterdir():
                print(f"  - {f.name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
