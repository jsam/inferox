use doclayout_yolo::DocLayoutYOLO;
use inferox_core::{Backend, Model, Tensor, TensorBuilder};
use inferox_tch::TchBackend;
use std::path::PathBuf;
use tch::Device;

fn get_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt")
}

const MODEL_INPUT_SIZE: usize = 1280;

#[tokio::test]
#[ignore]
async fn debug_model_output_format() {
    println!("=== Debugging DocLayout-YOLO Output Format ===\n");
    
    let model_path = get_model_path();
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load model");
    
    println!("1. Creating test input (1024×1024)...");
    let backend = TchBackend::cpu().expect("Failed to create backend");
    
    let input_tensor = backend
        .tensor_builder()
        .build_from_vec(
            vec![0.5f32; 1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE],
            &[1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        )
        .expect("Failed to create tensor");
    
    println!("   Input shape: {:?}\n", input_tensor.shape());
    
    println!("2. Running inference...");
    let output = model.forward(input_tensor).expect("Inference failed");
    
    println!("   Output shape: {:?}\n", output.shape());
    
    println!("3. Analyzing output tensor structure:");
    let shape = output.shape();
    let batch = shape[0];
    let features = shape[1];
    let boxes = shape[2];
    
    println!("   Batch size: {}", batch);
    println!("   Features per box: {}", features);
    println!("   Number of boxes: {}", boxes);
    println!();
    
    let output_vec: Vec<f32> = output.0.view([-1]).try_into().unwrap();
    
    println!("4. Sampling first 5 boxes:");
    for box_idx in 0..5.min(boxes) {
        println!("\n   Box {}:", box_idx);
        
        for feat_idx in 0..features {
            let value = output_vec[feat_idx * boxes + box_idx];
            
            let label = match feat_idx {
                0 => "x_center",
                1 => "y_center", 
                2 => "width",
                3 => "height",
                4..=13 => {
                    let class_names = [
                        "Caption", "Footnote", "Formula", "List-item",
                        "Page-footer", "Page-header", "Picture", "Section-header",
                        "Table", "Text",
                    ];
                    class_names[feat_idx - 4]
                }
                _ => "unknown",
            };
            
            if feat_idx < 4 {
                println!("     [{}] {}: {:.4}", feat_idx, label, value);
            } else {
                println!("     [{}] {}: {:.4}", feat_idx, label, value);
            }
        }
        
        let bbox_vals = [
            output_vec[0 * boxes + box_idx],
            output_vec[1 * boxes + box_idx],
            output_vec[2 * boxes + box_idx],
            output_vec[3 * boxes + box_idx],
        ];
        
        let class_scores: Vec<f32> = (4..14)
            .map(|i| output_vec[i * boxes + box_idx])
            .collect();
        
        let max_score = class_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let max_idx = class_scores.iter().position(|&x| x == max_score).unwrap();
        
        println!("     → BBox: x={:.1}, y={:.1}, w={:.1}, h={:.1}", 
                 bbox_vals[0], bbox_vals[1], bbox_vals[2], bbox_vals[3]);
        println!("     → Max class score: {:.4} (class {})", max_score, max_idx);
    }
    
    println!("\n5. Checking coordinate ranges:");
    let x_coords: Vec<f32> = (0..boxes).map(|i| output_vec[i]).collect();
    let y_coords: Vec<f32> = (0..boxes).map(|i| output_vec[boxes + i]).collect();
    let widths: Vec<f32> = (0..boxes).map(|i| output_vec[2 * boxes + i]).collect();
    let heights: Vec<f32> = (0..boxes).map(|i| output_vec[3 * boxes + i]).collect();
    
    let x_min = x_coords.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let x_max = x_coords.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_coords.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let y_max = y_coords.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let w_min = widths.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let w_max = widths.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let h_min = heights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let h_max = heights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("   X center: [{:.4}, {:.4}]", x_min, x_max);
    println!("   Y center: [{:.4}, {:.4}]", y_min, y_max);
    println!("   Width:    [{:.4}, {:.4}]", w_min, w_max);
    println!("   Height:   [{:.4}, {:.4}]", h_min, h_max);
    
    println!("\n6. Checking class score ranges:");
    for class_idx in 0..10 {
        let class_scores: Vec<f32> = (0..boxes)
            .map(|i| output_vec[(4 + class_idx) * boxes + i])
            .collect();
        
        let min_score = class_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_score = class_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_score = class_scores.iter().sum::<f32>() / class_scores.len() as f32;
        
        let class_names = [
            "Caption", "Footnote", "Formula", "List-item",
            "Page-footer", "Page-header", "Picture", "Section-header",
            "Table", "Text",
        ];
        
        println!("   {}: [{:.4}, {:.4}] (mean: {:.4})", 
                 class_names[class_idx], min_score, max_score, mean_score);
    }
    
    println!("\n7. Analysis Summary:");
    
    if x_max > 1024.0 || y_max > 1024.0 {
        println!("   ⚠️  Coordinates are in pixel space (not normalized)");
    } else if x_max <= 1.0 && y_max <= 1.0 {
        println!("   ✓ Coordinates appear to be normalized [0, 1]");
    } else {
        println!("   ⚠️  Coordinates in unknown range");
    }
    
    let mut overall_max_score = f32::NEG_INFINITY;
    for class_idx in 0..10 {
        for i in 0..boxes {
            let score = output_vec[(4 + class_idx) * boxes + i];
            if score > overall_max_score {
                overall_max_score = score;
            }
        }
    }
    
    if overall_max_score > 10.0 {
        println!("   ⚠️  Class scores are NOT probabilities (likely logits)");
        println!("   → Need to apply sigmoid/softmax activation");
    } else if overall_max_score <= 1.0 {
        println!("   ✓ Class scores appear to be probabilities");
    }
    
    println!("\n✅ Debug analysis complete!");
}

#[tokio::test]
#[ignore]
async fn compare_with_python_inference() {
    println!("=== Comparing Rust vs Python Inference ===\n");
    println!("This test requires running Python inference first.");
    println!("Run: cd models && python compare_inference.py");
}
