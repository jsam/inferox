# BERT-Candle Model Package

BERT model implementation using Candle backend for Inferox.

## Building the Complete Package

Simply run:

```bash
cargo build --release -p bert-candle
```

On first build, it compiles the architecture. On second build, it automatically:
1. Downloads config.json + model.safetensors from HuggingFace (specified in `model.toml`)
2. Assembles complete package at `target/mlpkg/bert-candle/`

The package is ready to use - everything is in the `target/` directory (gitignored).

## Configuration

Edit `model.toml` to use a different HuggingFace model:
```toml
repo_id = "bert-large-uncased"
backend = "candle"
```

This will create `target/mlpkg/bert-candle/` containing:
```
mlpkg/
├── model_info.json          # Model metadata
├── metadata.json            # Package metadata
└── backends/
    └── candle/
        ├── config.json      # HuggingFace config
        ├── model.safetensors # Model weights
        └── libmodel.dylib   # Compiled architecture
```

## Environment Variables

- `BERT_REPO_ID` - HuggingFace repository to download (default: `bert-base-uncased`)

Example:
```bash
BERT_REPO_ID=bert-large-uncased cargo run --bin bert-candle-package --features packaging --release
```

## Using the Package

```rust
use inferox_mlpkg::{BackendType, PackageManager};

let manager = PackageManager::new(cache_dir)?;

// Load from mlpkg directory
let package_path = PathBuf::from("target/mlpkg/bert-candle");
let package = manager.load_package(&package_path)?;

// Load model (backend determined from model.toml)
let loaded_model = manager.load_model(&package)?;
let model = loaded_model.as_candle().expect("Expected Candle model");
let output = model.forward(input)?;
```

## Architecture

The BERT implementation:
- **`BertModelWrapper`** - Implements `Model` trait from `inferox-core`
- **`create_model()`** - Factory function loaded via `libloading`
- Reads config and weights from `INFEROX_PACKAGE_DIR` at runtime
- Uses `candle-transformers::models::bert::BertModel` internally
