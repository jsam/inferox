# MLP Example

Demonstrates the **two-pillar architecture** of Inferox: compiling model architectures into shared libraries (`.so`/`.dylib`) and loading them safely into the engine via expected trait interfaces.

## Architecture

This example includes two MLP models with different sizes:

- **Classifier**: 10 → 8 → 8 → 3 (for demonstration purposes)
- **Small**: 5 → 4 → 4 → 2 (for demonstration purposes)

Each model is compiled as a separate shared library.

## Structure

```
mlp/
├── src/
│   ├── lib.rs                    # MLP architecture implementation
│   └── main.rs                   # Engine that loads .so files
├── models/
│   ├── classifier/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs           # Exports create_model() -> Box<dyn Model>
│   └── small/
│       ├── Cargo.toml
│       └── src/lib.rs           # Exports create_model() -> Box<dyn Model>
```

## Building

```bash
# Build model libraries (.dylib/.so files)
cargo build --release -p mlp-classifier -p mlp-small

# Build the engine
cargo build --release -p mlp
```

This creates:
- `../../target/release/libmlp_classifier.dylib` (or `.so` on Linux)
- `../../target/release/libmlp_small.dylib` (or `.so` on Linux)
- `../../target/release/mlp` (the engine binary)

Note: Cargo workspaces place all build artifacts in the workspace root's `target/` directory.

## Running

```bash
cargo run --bin mlp --release -- ../../target/release/libmlp_classifier.dylib ../../target/release/libmlp_small.dylib
```

## Example Output

```
Inferox MLP Engine
==================

✓ Created CPU backend

Loading model from: target/release/libmlp_classifier.dylib
✓ Registered 'classifier' - Multi-Layer Perceptron (10 → 8 → 8 → 3)

Loading model from: target/release/libmlp_small.dylib
✓ Registered 'small' - Multi-Layer Perceptron (5 → 4 → 4 → 2)

2 models loaded

Available models:
  - classifier v1.0.0: Multi-Layer Perceptron (10 → 8 → 8 → 3)
  - small v1.0.0: Multi-Layer Perceptron (5 → 4 → 4 → 2)

Running test inference on all models:
  classifier -> output shape: [1, 3]
  small -> output shape: [1, 2]

✓ All models working!
```

## Architecture Flow

```
┌─────────────────────────────────────────────┐
│  Model Library (libmlp_classifier.dylib)    │
│  ┌──────────────────────────────────────┐   │
│  │ #[no_mangle]                         │   │
│  │ pub fn create_model()                │   │
│  │   -> Box<dyn Model>                  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  Engine (loads via libloading)              │
│  ┌──────────────────────────────────────┐   │
│  │ let lib = Library::new(path)         │   │
│  │ let factory = lib.get("create_model")│   │
│  │ let model: Box<dyn Model> = factory()│   │
│  │ engine.register_boxed_model(model)   │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  InferoxEngine manages all models           │
│  - Type-safe trait interface                │
│  - RAII memory management                   │
│  - No unsafe in user code                   │
└─────────────────────────────────────────────┘
```

## How It Works

### 1. Model Libraries (`models/*/src/lib.rs`)

Each model is a separate crate that exports a `create_model()` function:

```rust
#[no_mangle]
pub fn create_model() -> Box<dyn Model<Backend = CandleBackend, Input = CandleTensor, Output = CandleTensor>> {
    let builder = CandleModelBuilder::new(Device::Cpu);
    let model = MLP::new("classifier", 10, 8, 3, builder.var_builder())
        .expect("Failed to create classifier model");
    Box::new(model)
}
```

### 2. Engine (`src/main.rs`)

The engine loads these libraries at runtime using `libloading`:

```rust
unsafe {
    let lib = Library::new(path)?;
    let factory: Symbol<ModelFactory> = lib.get(b"create_model")?;
    let model = factory();  // Returns Box<dyn Model>
    engine.register_boxed_model(model);
}
```

### 3. Shared Trait Interface

All models implement the same `Model` trait from `inferox-core`, ensuring:
- ✅ **Type safety**: Trait bounds enforced at compile time
- ✅ **Memory safety**: Rust ownership and RAII
- ✅ **No unsafe in user code**: Only `libloading` uses unsafe
- ✅ **Hot reloading**: Replace `.dylib` files without recompiling engine

## Key Benefits

1. **Separation of Concerns**: Model compilation separate from runtime
2. **Dynamic Loading**: Load/unload models at runtime
3. **Safe Rust Interface**: No manual FFI, no C headers needed
4. **Multiple Models**: Engine manages multiple models simultaneously
5. **Practical Sizes**: Small input sizes (5-10) for easy CLI testing

## Testing

```bash
cargo test -p mlp
```
