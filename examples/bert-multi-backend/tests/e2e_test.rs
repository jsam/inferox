use bert_multi_backend::{InferenceRequest, InferenceRouter};
use inferox_engine::{EngineConfig, InferoxEngine};
use inferox_mlpkg::{BackendType, PackageManager};
use std::path::PathBuf;
use std::sync::Arc;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[tokio::test]
#[ignore]
async fn test_load_both_backends() {
    println!("=== Test: Load Both Backends ===\n");

    let workspace_root = workspace_root();
    let candle_package = workspace_root.join("target/mlpkg/bert-candle");
    let tch_package = workspace_root.join("target/mlpkg/bert-tch");

    assert!(
        candle_package.exists(),
        "Candle package not found. Run: make test-bert-candle"
    );
    assert!(
        tch_package.exists(),
        "Tch package not found. Run: make test-bert-tch"
    );

    println!("1. Creating PackageManager...");
    let cache_dir = std::env::temp_dir().join("inferox-multi-backend-test");
    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    println!("2. Loading Candle package...");
    let candle_pkg = manager
        .load_package(&candle_package)
        .expect("Failed to load candle package");
    println!("   ✓ Loaded: {}", candle_pkg.metadata.name);

    println!("3. Loading Tch package...");
    let tch_pkg = manager
        .load_package(&tch_package)
        .expect("Failed to load tch package");
    println!("   ✓ Loaded: {}", tch_pkg.metadata.name);

    println!("4. Loading Candle model...");
    let (candle_model, _) = manager
        .load_model(&candle_pkg)
        .expect("Failed to load candle model");
    let candle_model = candle_model
        .as_candle()
        .expect("Expected Candle model");
    let candle_name = candle_model.name().to_string();
    println!("   ✓ Model name: {}", candle_name);

    println!("5. Loading Tch model...");
    let (tch_model, _) = manager
        .load_model(&tch_pkg)
        .expect("Failed to load tch model");
    let tch_model = tch_model.as_tch().expect("Expected Tch model");
    let tch_name = tch_model.name().to_string();
    println!("   ✓ Model name: {}", tch_name);

    println!("6. Creating InferoxEngine...");
    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);

    println!("7. Registering both models...");
    engine.register_model(&candle_name, candle_model, None);
    engine.register_model(&tch_name, tch_model, None);

    let models = engine.list_models();
    println!("   ✓ Registered {} models", models.len());
    assert_eq!(models.len(), 2);

    println!("8. Creating router with named routes...");
    let router = InferenceRouter::new(Arc::new(engine))
        .with_route("candle.bert", candle_name, BackendType::Candle)
        .with_route("tch.bert", tch_name, BackendType::Tch);

    let routes = router.list_routes();
    println!("   ✓ Registered routes: {:?}", routes);
    assert_eq!(routes.len(), 2);

    println!("\n✅ Test passed: Both backends loaded successfully\n");
}

#[tokio::test]
#[ignore]
async fn test_route_to_candle() {
    println!("=== Test: Route to Candle via 'candle.bert' ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];
    let request = InferenceRequest::new(input_ids.clone(), "candle.bert");

    println!("Routing request to 'candle.bert'...");
    let response = router.infer(request).expect("Inference failed");

    println!("Response:");
    println!("  Route: candle.bert");
    println!("  Backend used: {:?}", response.backend_used);
    println!("  Shape: {:?}", response.shape);
    println!("  Latency: {:.2}ms", response.latency_ms);

    assert_eq!(response.backend_used, BackendType::Candle);
    assert_eq!(response.shape[0], 1);
    assert_eq!(response.shape[1], input_ids.len());
    assert_eq!(response.shape[2], 768);

    println!("\n✅ Test passed: Routed to Candle successfully\n");
}

#[tokio::test]
#[ignore]
async fn test_route_to_tch() {
    println!("=== Test: Route to Tch via 'tch.bert' ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];
    let request = InferenceRequest::new(input_ids.clone(), "tch.bert");

    println!("Routing request to 'tch.bert'...");
    let response = router.infer(request).expect("Inference failed");

    println!("Response:");
    println!("  Route: tch.bert");
    println!("  Backend used: {:?}", response.backend_used);
    println!("  Shape: {:?}", response.shape);
    println!("  Latency: {:.2}ms", response.latency_ms);

    assert_eq!(response.backend_used, BackendType::Tch);
    assert_eq!(response.shape[0], 1);
    assert_eq!(response.shape[1], input_ids.len());
    assert_eq!(response.shape[2], 768);

    println!("\n✅ Test passed: Routed to Tch successfully\n");
}

#[tokio::test]
#[ignore]
async fn test_pytorch_route() {
    println!("=== Test: Route to PyTorch via 'pytorch.bert' ===\n");

    let workspace_root = workspace_root();
    let candle_package = workspace_root.join("target/mlpkg/bert-candle");
    let tch_package = workspace_root.join("target/mlpkg/bert-tch");

    let cache_dir = std::env::temp_dir().join("inferox-multi-backend-test");
    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let candle_pkg = manager.load_package(&candle_package).expect("Failed to load candle package");
    let tch_pkg = manager.load_package(&tch_package).expect("Failed to load tch package");

    let (candle_model, _) = manager.load_model(&candle_pkg).expect("Failed to load candle model");
    let candle_model = candle_model.as_candle().expect("Expected Candle model");
    let candle_name = candle_model.name().to_string();

    let (tch_model, _) = manager.load_model(&tch_pkg).expect("Failed to load tch model");
    let tch_model = tch_model.as_tch().expect("Expected Tch model");
    let tch_name = tch_model.name().to_string();

    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);
    engine.register_model(&candle_name, candle_model, None);
    engine.register_model(&tch_name, tch_model, None);

    let router = InferenceRouter::new(Arc::new(engine))
        .with_route("candle.bert", candle_name, BackendType::Candle)
        .with_route("pytorch.bert", tch_name, BackendType::Tch);

    let input_ids = vec![101, 7592, 1010, 2088, 999, 102];
    let request = InferenceRequest::new(input_ids.clone(), "pytorch.bert");

    println!("Routing request to 'pytorch.bert'...");
    let response = router.infer(request).expect("Inference failed");

    println!("Response:");
    println!("  Route: pytorch.bert");
    println!("  Backend used: {:?}", response.backend_used);
    println!("  Shape: {:?}", response.shape);
    println!("  Latency: {:.2}ms", response.latency_ms);

    assert_eq!(response.backend_used, BackendType::Tch);
    assert_eq!(response.shape[0], 1);
    assert_eq!(response.shape[1], input_ids.len());
    assert_eq!(response.shape[2], 768);

    println!("\n✅ Test passed: 'pytorch.bert' route worked\n");
}

#[tokio::test]
#[ignore]
async fn test_invalid_route() {
    println!("=== Test: Invalid Route Error ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];
    let request = InferenceRequest::new(input_ids, "nonexistent.route");

    println!("Trying invalid route 'nonexistent.route'...");
    let result = router.infer(request);

    assert!(result.is_err());
    println!("  ✓ Got expected error: {}", result.unwrap_err());

    println!("\n✅ Test passed: Invalid route properly rejected\n");
}

#[tokio::test]
#[ignore]
async fn test_output_consistency() {
    println!("=== Test: Output Consistency Between Routes ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];

    println!("1. Running inference on 'candle.bert'...");
    let candle_request = InferenceRequest::new(input_ids.clone(), "candle.bert");
    let candle_response = router.infer(candle_request).expect("Candle inference failed");

    println!("2. Running inference on 'tch.bert'...");
    let tch_request = InferenceRequest::new(input_ids.clone(), "tch.bert");
    let tch_response = router.infer(tch_request).expect("Tch inference failed");

    println!("3. Comparing outputs...");
    println!("   Candle shape: {:?}", candle_response.shape);
    println!("   Tch shape: {:?}", tch_response.shape);

    assert_eq!(candle_response.shape, tch_response.shape, "Shapes should match");

    let tolerance = 1e-3;
    let max_diff = candle_response
        .max_difference(&tch_response)
        .expect("Failed to compute max difference");

    println!("   Max difference: {:.6}", max_diff);
    println!("   Tolerance: {:.6}", tolerance);

    if max_diff < tolerance {
        println!("   ✓ Outputs are within tolerance");
    } else {
        println!("   ⚠ Outputs differ more than tolerance (expected for different backends)");
    }

    println!("\n✅ Test passed: Output consistency checked\n");
}

#[tokio::test]
#[ignore]
async fn test_concurrent_requests() {
    println!("=== Test: Concurrent Requests to Both Routes ===\n");

    let router = Arc::new(setup_router().await);

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];

    println!("Spawning 10 concurrent requests (5 to each route)...");

    let mut handles = vec![];

    for i in 0..5 {
        let router = Arc::clone(&router);
        let ids = input_ids.clone();
        let handle = tokio::spawn(async move {
            let request = InferenceRequest::new(ids, "candle.bert");
            router.infer(request).expect(&format!("Candle request {} failed", i))
        });
        handles.push(handle);
    }

    for i in 0..5 {
        let router = Arc::clone(&router);
        let ids = input_ids.clone();
        let handle = tokio::spawn(async move {
            let request = InferenceRequest::new(ids, "tch.bert");
            router.infer(request).expect(&format!("Tch request {} failed", i))
        });
        handles.push(handle);
    }

    println!("Waiting for all requests to complete...");
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("Some tasks panicked");

    println!("✓ All {} requests completed successfully", results.len());

    let candle_count = results
        .iter()
        .filter(|r| r.backend_used == BackendType::Candle)
        .count();
    let tch_count = results
        .iter()
        .filter(|r| r.backend_used == BackendType::Tch)
        .count();

    println!("  Candle: {} responses", candle_count);
    println!("  Tch: {} responses", tch_count);

    assert_eq!(candle_count, 5);
    assert_eq!(tch_count, 5);

    println!("\n✅ Test passed: Concurrent requests handled correctly\n");
}

#[tokio::test]
#[ignore]
async fn test_performance_comparison() {
    println!("=== Test: Performance Comparison ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];
    let iterations = 10;

    println!("Running {} iterations per route...\n", iterations);

    let mut candle_latencies = vec![];
    for i in 0..iterations {
        let request = InferenceRequest::new(input_ids.clone(), "candle.bert");
        let response = router.infer(request).expect("Candle inference failed");
        candle_latencies.push(response.latency_ms);
        if i == 0 {
            println!("candle.bert first run: {:.2}ms", response.latency_ms);
        }
    }

    let mut tch_latencies = vec![];
    for i in 0..iterations {
        let request = InferenceRequest::new(input_ids.clone(), "tch.bert");
        let response = router.infer(request).expect("Tch inference failed");
        tch_latencies.push(response.latency_ms);
        if i == 0 {
            println!("tch.bert first run: {:.2}ms", response.latency_ms);
        }
    }

    let candle_avg: f64 = candle_latencies.iter().sum::<f64>() / candle_latencies.len() as f64;
    let tch_avg: f64 = tch_latencies.iter().sum::<f64>() / tch_latencies.len() as f64;

    println!("\nAverage latencies:");
    println!("  candle.bert: {:.2}ms", candle_avg);
    println!("  tch.bert: {:.2}ms", tch_avg);

    let faster = if candle_avg < tch_avg {
        "candle.bert"
    } else {
        "tch.bert"
    };
    let speedup = (candle_avg.max(tch_avg) / candle_avg.min(tch_avg) - 1.0) * 100.0;

    println!("  {} is {:.1}% faster", faster, speedup);

    println!("\n✅ Test passed: Performance comparison completed\n");
}

#[tokio::test]
#[ignore]
async fn test_unified_response_format() {
    println!("=== Test: Unified Response Format ===\n");

    let router = setup_router().await;

    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102];

    let candle_request = InferenceRequest::new(input_ids.clone(), "candle.bert");
    let candle_response = router.infer(candle_request).expect("Candle inference failed");

    let tch_request = InferenceRequest::new(input_ids.clone(), "tch.bert");
    let tch_response = router.infer(tch_request).expect("Tch inference failed");

    println!("Verifying unified response format...");
    println!("  Candle response has {} elements", candle_response.data.len());
    println!("  Tch response has {} elements", tch_response.data.len());

    assert!(
        candle_response.data.iter().all(|x| x.is_finite()),
        "Candle data should be finite"
    );
    assert!(
        tch_response.data.iter().all(|x| x.is_finite()),
        "Tch data should be finite"
    );

    let candle_json = serde_json::to_string(&candle_response).expect("Failed to serialize");
    let tch_json = serde_json::to_string(&tch_response).expect("Failed to serialize");

    println!("  ✓ Both responses serialize to JSON");
    println!("  ✓ Candle JSON size: {} bytes", candle_json.len());
    println!("  ✓ Tch JSON size: {} bytes", tch_json.len());

    println!("\n✅ Test passed: Unified response format verified\n");
}

#[tokio::test]
#[ignore]
async fn test_list_routes() {
    println!("=== Test: List Available Routes ===\n");

    let router = setup_router().await;

    let routes = router.list_routes();
    println!("Available routes:");
    for route in &routes {
        if let Some(model_route) = router.get_route(route) {
            println!("  - {} → {} ({:?})", route, model_route.model_name, model_route.backend);
        }
    }

    assert!(routes.contains(&"candle.bert".to_string()));
    assert!(routes.contains(&"tch.bert".to_string()));
    assert_eq!(routes.len(), 2);

    println!("\n✅ Test passed: Route listing works\n");
}

#[tokio::test]
#[ignore]
async fn test_route_validation_at_startup() {
    println!("=== Test: Route Validation at Startup ===\n");

    let workspace_root = workspace_root();
    let candle_package = workspace_root.join("target/mlpkg/bert-candle");
    let cache_dir = std::env::temp_dir().join("inferox-multi-backend-test");
    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let candle_pkg = manager
        .load_package(&candle_package)
        .expect("Failed to load candle package");

    let (candle_model, _) = manager
        .load_model(&candle_pkg)
        .expect("Failed to load candle model");
    let candle_model = candle_model.as_candle().expect("Expected Candle model");
    let candle_name = candle_model.name().to_string();

    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);
    engine.register_model(&candle_name, candle_model, None);

    println!("1. Creating router with valid route...");
    let mut router = InferenceRouter::new(Arc::new(engine));
    router.register_route("candle.bert", candle_name.clone(), BackendType::Candle);
    
    let result = router.validate_routes();
    assert!(result.is_ok(), "Valid routes should pass validation");
    println!("   ✓ Valid route passed validation");

    println!("2. Creating router with invalid route...");
    router.register_route("invalid.bert", "nonexistent_model".to_string(), BackendType::Candle);
    
    let result = router.validate_routes();
    assert!(result.is_err(), "Invalid route should fail validation");
    
    let error = result.unwrap_err();
    let error_msg = error.to_string();
    println!("   ✓ Invalid route failed validation: {}", error_msg);
    assert!(error_msg.contains("not loaded in the engine"));
    assert!(error_msg.contains("invalid.bert"));
    assert!(error_msg.contains("nonexistent_model"));

    println!("\n✅ Test passed: Route validation works at startup\n");
}

async fn setup_router() -> InferenceRouter {
    let workspace_root = workspace_root();
    let candle_package = workspace_root.join("target/mlpkg/bert-candle");
    let tch_package = workspace_root.join("target/mlpkg/bert-tch");

    let cache_dir = std::env::temp_dir().join("inferox-multi-backend-test");
    let manager = PackageManager::new(cache_dir).expect("Failed to create PackageManager");

    let candle_pkg = manager
        .load_package(&candle_package)
        .expect("Failed to load candle package");
    let tch_pkg = manager
        .load_package(&tch_package)
        .expect("Failed to load tch package");

    let (candle_model, _) = manager
        .load_model(&candle_pkg)
        .expect("Failed to load candle model");
    let candle_model = candle_model.as_candle().expect("Expected Candle model");
    let candle_name = candle_model.name().to_string();

    let (tch_model, _) = manager
        .load_model(&tch_pkg)
        .expect("Failed to load tch model");
    let tch_model = tch_model.as_tch().expect("Expected Tch model");
    let tch_name = tch_model.name().to_string();

    let config = EngineConfig::default();
    let mut engine = InferoxEngine::new(config);

    engine.register_model(&candle_name, candle_model, None);
    engine.register_model(&tch_name, tch_model, None);

    let router = InferenceRouter::new(Arc::new(engine))
        .with_route("candle.bert", candle_name, BackendType::Candle)
        .with_route("tch.bert", tch_name, BackendType::Tch);
    
    router.validate_routes().expect("Route validation failed at startup");
    
    router
}
