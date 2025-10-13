#!/usr/bin/env python3
"""
Convert PDF pages to images for DocLayout-YOLO testing.
"""

import os
from pathlib import Path
from pdf2image import convert_from_path

def prepare_samples():
    samples_dir = Path(__file__).parent
    pdf_path = samples_dir / "2410.12628v1.pdf"
    inputs_dir = samples_dir / "inputs"
    
    inputs_dir.mkdir(exist_ok=True)
    
    print(f"Converting {pdf_path.name} to images...")
    
    # Convert first 3 pages to images
    images = convert_from_path(pdf_path, first_page=1, last_page=3, dpi=150)
    
    for i, image in enumerate(images, 1):
        output_path = inputs_dir / f"page_{i}.png"
        image.save(output_path, 'PNG')
        print(f"  Saved: {output_path.name} ({image.size[0]}x{image.size[1]})")
    
    print(f"\n✓ Converted {len(images)} pages to {inputs_dir}/")
    print(f"  Ready for inference!")

if __name__ == "__main__":
    prepare_samples()
