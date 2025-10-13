use doclayout_yolo::DocLayoutYOLO;
use ab_glyph::{FontRef, PxScale};
use image::Rgb;
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use inferox_core::{Backend, Model, Tensor, TensorBuilder};
use inferox_engine::{EngineConfig, InferoxEngine};
use inferox_tch::TchBackend;
use std::path::{Path, PathBuf};
use tch::Device;

const CLASS_NAMES: &[&str] = &[
    "title",
    "plain text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
];

const CLASS_THRESHOLDS: &[f32] = &[
    0.5,  // title
    0.4,  // plain text (common)
    0.5,  // abandon
    0.6,  // figure (require high confidence)
    0.5,  // figure_caption
    0.6,  // table (require high confidence)
    0.5,  // table_caption
    0.5,  // table_footnote
    0.6,  // isolate_formula (require high confidence)
    0.5,  // formula_caption
];

const CLASS_COLORS: &[(u8, u8, u8)] = &[
    (255, 0, 0),     
    (0, 255, 0),     
    (0, 0, 255),     
    (255, 255, 0),   
    (255, 0, 255),   
    (0, 255, 255),   
    (128, 0, 0),     
    (0, 128, 0),     
    (0, 0, 128),     
    (128, 128, 0),   
    (128, 0, 128),   
];

#[derive(Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
    class_id: usize,
}

fn calculate_iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    
    if x2 < x1 || y2 < y1 {
        return 0.0;
    }
    
    let intersection = (x2 - x1) * (y2 - y1);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - intersection;
    
    if union <= 0.0 {
        return 0.0;
    }
    
    intersection / union
}

fn apply_nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut keep = Vec::new();
    let mut sorted = detections.to_vec();
    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    while !sorted.is_empty() {
        let best = sorted.remove(0);
        keep.push(best.clone());
        
        sorted.retain(|det| {
            if det.class_id != best.class_id {
                return true;
            }
            
            let iou = calculate_iou(&best, det);
            iou < iou_threshold
        });
    }
    
    keep
}

fn filter_invalid_detections(detections: &[Detection], image_size: (u32, u32)) -> Vec<Detection> {
    detections.iter().filter(|det| {
        let width = det.x2 - det.x1;
        let height = det.y2 - det.y1;
        let area = width * height;
        let img_area = (image_size.0 * image_size.1) as f32;
        
        width > 10.0 &&
        height > 10.0 &&
        area < img_area * 0.95 &&
        width / height < 50.0 &&
        height / width < 50.0
    }).cloned().collect()
}

fn get_samples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("samples")
}

fn get_inputs_dir() -> PathBuf {
    get_samples_dir().join("inputs")
}

fn get_outputs_dir() -> PathBuf {
    get_samples_dir().join("outputs")
}

fn get_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/doclayout_yolo_docstructbench_imgsz1280_2501_torchscript.pt")
}

fn load_and_preprocess_image(image_path: &Path, target_size: usize) -> Result<(Vec<f32>, u32, u32), Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let resized = img.resize_exact(
        target_size as u32,
        target_size as u32,
        image::imageops::FilterType::Lanczos3,
    );
    
    let rgb_img = resized.to_rgb8();
    
    let mut tensor_data = Vec::with_capacity(3 * target_size * target_size);
    
    for c in 0..3 {
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                let value = pixel[c] as f32 / 255.0;
                tensor_data.push(value);
            }
        }
    }
    
    Ok((tensor_data, orig_width, orig_height))
}

fn parse_detections(
    output: &inferox_tch::TchTensor,
    orig_width: u32,
    orig_height: u32,
    model_size: usize,
    _conf_threshold: f32,
) -> Vec<Detection> {
    let shape = output.shape();
    println!("   Raw output shape: {:?}", shape);
    
    if shape.len() != 3 || shape[0] != 1 {
        println!("   Warning: Unexpected output shape, expected [1, num_features, num_boxes]");
        return Vec::new();
    }
    
    let num_features = shape[1];
    let num_boxes = shape[2];
    
    println!("   Parsing {} features × {} boxes", num_features, num_boxes);
    println!("   Format: [x, y, w, h, class0, class1, ..., class10] = 14 features");
    
    let mut detections = Vec::new();
    
    let output_vec: Vec<f32> = output.0.view([-1]).try_into().unwrap_or_default();
    
    for box_idx in 0..num_boxes {
        let base_idx = box_idx;
        let stride = num_boxes;
        
        if base_idx + 13 * stride >= output_vec.len() {
            break;
        }
        
        let x_center = output_vec[base_idx + 0 * stride];
        let y_center = output_vec[base_idx + 1 * stride];
        let width = output_vec[base_idx + 2 * stride];
        let height = output_vec[base_idx + 3 * stride];
        
        let mut class_scores = Vec::new();
        for class_idx in 0..10 {
            class_scores.push(output_vec[base_idx + (4 + class_idx) * stride]);
        }
        
        let (class_id, &max_score) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        let threshold = CLASS_THRESHOLDS.get(class_id).copied().unwrap_or(0.5);
        if max_score < threshold {
            continue;
        }
        
        let x1 = ((x_center - width / 2.0) * orig_width as f32 / model_size as f32).max(0.0);
        let y1 = ((y_center - height / 2.0) * orig_height as f32 / model_size as f32).max(0.0);
        let x2 = ((x_center + width / 2.0) * orig_width as f32 / model_size as f32).min(orig_width as f32);
        let y2 = ((y_center + height / 2.0) * orig_height as f32 / model_size as f32).min(orig_height as f32);
        
        if x2 > x1 && y2 > y1 && width > 0.0 && height > 0.0 {
            detections.push(Detection {
                x1,
                y1,
                x2,
                y2,
                confidence: max_score,
                class_id,
            });
        }
    }
    
    let filtered = filter_invalid_detections(&detections, (orig_width, orig_height));
    
    let nms_detections = apply_nms(&filtered, 0.5);
    
    nms_detections
}

fn draw_detections(
    image_path: &Path,
    detections: &[Detection],
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut img = image::open(image_path)?.to_rgb8();
    
    let font_data = include_bytes!("../../../assets/DejaVuSans.ttf");
    let font = FontRef::try_from_slice(font_data).map_err(|_| "Failed to load font")?;
    let scale = PxScale::from(20.0);
    
    for detection in detections {
        let color = CLASS_COLORS[detection.class_id % CLASS_COLORS.len()];
        let rgb_color = Rgb([color.0, color.1, color.2]);
        
        let rect = Rect::at(detection.x1 as i32, detection.y1 as i32)
            .of_size(
                (detection.x2 - detection.x1) as u32,
                (detection.y2 - detection.y1) as u32,
            );
        
        draw_hollow_rect_mut(&mut img, rect, rgb_color);
        
        for offset in [-1, 0, 1] {
            draw_hollow_rect_mut(
                &mut img,
                Rect::at(rect.left() + offset, rect.top() + offset).of_size(rect.width(), rect.height()),
                rgb_color,
            );
        }
        
        let class_name = CLASS_NAMES.get(detection.class_id).unwrap_or(&"Unknown");
        let label = format!("{}: {:.2}", class_name, detection.confidence);
        
        let text_y = (detection.y1 as i32 - 25).max(0);
        draw_text_mut(&mut img, rgb_color, detection.x1 as i32, text_y, scale, &font, &label);
    }
    
    img.save(output_path)?;
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_visual_inference_pipeline() {
    let inputs_dir = get_inputs_dir();
    let outputs_dir = get_outputs_dir();
    
    if !inputs_dir.exists() {
        panic!(
            "Input images not found at {:?}. Run: cd samples && python prepare_samples.py",
            inputs_dir
        );
    }
    
    std::fs::create_dir_all(&outputs_dir).expect("Failed to create outputs directory");
    
    println!("=== DocLayout-YOLO Visual E2E Test ===\n");
    println!("1. Setting up Inferox Engine...");
    
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);
    println!("   ✓ Engine created\n");
    
    println!("2. Loading DocLayout-YOLO model...");
    let model_path = get_model_path();
    
    if !model_path.exists() {
        panic!(
            "Model not found at {:?}. Please export the model first.",
            model_path
        );
    }
    
    let model = DocLayoutYOLO::from_pretrained(&model_path, Device::Cpu)
        .expect("Failed to load DocLayout-YOLO model");
    
    let model_name = model.name().to_string();
    println!("   ✓ Model loaded: {}\n", model_name);
    
    println!("3. Registering model with engine...");
    engine.register_model(&model_name, Box::new(model), None);
    println!("   ✓ Model registered\n");
    
    let image_files: Vec<_> = std::fs::read_dir(&inputs_dir)
        .expect("Failed to read inputs directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().and_then(|s| s.to_str()) == Some("png")
        })
        .collect();
    
    println!("4. Processing {} images...\n", image_files.len());
    
    let backend = TchBackend::cpu().expect("Failed to create backend");
    let conf_threshold = 0.5;
    let model_size = if model_path.to_string_lossy().contains("1280") { 1280 } else { 1024 };
    
    for (idx, entry) in image_files.iter().enumerate() {
        let image_path = entry.path();
        let image_name = image_path.file_name().unwrap().to_string_lossy();
        
        println!("   [{}] Processing: {}", idx + 1, image_name);
        
        let (tensor_data, orig_width, orig_height) = load_and_preprocess_image(&image_path, model_size)
            .expect("Failed to preprocess image");
        
        println!("       Original size: {}×{}", orig_width, orig_height);
        println!("       Model input: {}×{}", model_size, model_size);
        
        let input_tensor = backend
            .tensor_builder()
            .build_from_vec(tensor_data, &[1, 3, model_size, model_size])
            .expect("Failed to create tensor");
        
        let output = engine
            .infer_typed::<TchBackend>(&model_name, input_tensor)
            .expect("Inference failed");
        
        println!("       Inference completed, output shape: {:?}", output.shape());
        
        let detections = parse_detections(&output, orig_width, orig_height, model_size, conf_threshold);
        
        println!("       Found {} detections (after NMS & filtering)", detections.len());
        
        for (i, det) in detections.iter().take(5).enumerate() {
            let class_name = CLASS_NAMES.get(det.class_id).unwrap_or(&"Unknown");
            println!(
                "         {}. {}: {:.2}% at ({:.0}, {:.0}, {:.0}, {:.0})",
                i + 1,
                class_name,
                det.confidence * 100.0,
                det.x1, det.y1, det.x2, det.y2
            );
        }
        
        if detections.len() > 5 {
            println!("         ... and {} more", detections.len() - 5);
        }
        
        let output_path = outputs_dir.join(&*image_name);
        draw_detections(&image_path, &detections, &output_path)
            .expect("Failed to draw detections");
        
        println!("       ✓ Saved visualization: {}\n", output_path.display());
    }
    
    println!("✅ Visual E2E test completed!");
    println!("   Input images: {}", inputs_dir.display());
    println!("   Output visualizations: {}", outputs_dir.display());
    println!("   Model: DocLayout-YOLO via TorchScript");
    println!("   Engine: Inferox with TchBackend");
}
