# Inferox Makefile
# Provides convenient commands for testing, building, and development

.PHONY: help build build-release test test-quick lint lint-quick pre-commit clean doc examples models run-example

# Default target
help:
	@echo "Inferox Development Commands"
	@echo "============================="
	@echo ""
	@echo "Building:"
	@echo "  build           - Build all crates in debug mode"
	@echo "  build-release   - Build all crates in release mode"
	@echo "  models          - Build all model libraries (.dylib/.so)"
	@echo "  examples        - Build all example binaries"
	@echo ""
	@echo "Testing:"
	@echo "  test            - Run tests + quick lint (recommended)"
	@echo "  test-quick      - Run tests only (faster)"
	@echo "  test-core       - Run core library tests only"
	@echo "  test-candle     - Run Candle backend tests only"
	@echo "  test-engine     - Run engine tests only"
	@echo "  test-examples   - Run example tests"
	@echo ""
	@echo "Development:"
	@echo "  pre-commit      - Quick pre-commit checks (format + lint + test)"
	@echo "  lint            - Run full clippy linter on all targets"
	@echo "  lint-quick      - Run quick clippy check (lib + bins only)"
	@echo "  format          - Format code with rustfmt"
	@echo "  format-check    - Check if code is formatted (CI)"
	@echo "  check           - Check code compiles without building"
	@echo "  clean           - Clean build artifacts"
	@echo ""
	@echo "Running:"
	@echo "  run-example     - Run MLP example with model libraries"
	@echo ""
	@echo "Documentation:"
	@echo "  doc             - Generate and open documentation"
	@echo "  doc-private     - Generate documentation including private items"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci              - Run all CI checks (lint + test)"
	@echo "  ci-test         - Run CI tests"
	@echo "  ci-lint         - Run CI linting"

# Building commands
build:
	@echo "Building all crates..."
	cargo build

build-release:
	@echo "Building all crates in release mode..."
	cargo build --release

models:
	@echo "Building model libraries..."
	cargo build --release -p mlp-classifier -p mlp-small
	@echo ""
	@echo "✅ Model libraries built:"
	@echo "  - target/release/libmlp_classifier.dylib (or .so on Linux)"
	@echo "  - target/release/libmlp_small.dylib (or .so on Linux)"

examples:
	@echo "Building all examples..."
	cargo build -p mlp
	@echo "✅ Examples built"

# Testing commands
test: test-quick lint-quick
	@echo ""
	@echo "✅ All tests and checks passed!"

test-quick:
	@echo "Running all tests..."
	cargo test

test-core:
	@echo "Running core library tests..."
	cargo test -p inferox-core

test-candle:
	@echo "Running Candle backend tests..."
	cargo test -p inferox-candle

test-engine:
	@echo "Running engine tests..."
	cargo test -p inferox-engine

test-examples:
	@echo "Running example tests..."
	cargo test -p mlp

lint-quick:
	@echo ""
	@echo "Running quick lint check..."
	@cargo clippy --lib --bins -- -D warnings || (echo "❌ Clippy failed! Run 'cargo clippy --fix' to auto-fix." && exit 1)
	@echo "✅ Lint check passed"

# Development commands
lint:
	@echo "Running clippy linter..."
	cargo clippy --all-targets --all-features -- -D warnings

format:
	@echo "Formatting code..."
	cargo fmt

format-check:
	@echo "Checking code formatting..."
	cargo fmt --check

check:
	@echo "Checking code compiles..."
	cargo check --all-targets

clean:
	@echo "Cleaning build artifacts..."
	cargo clean

# Running commands
run-example: models
	@echo ""
	@echo "Running MLP example..."
	cargo run --bin mlp --release -- target/release/libmlp_classifier.dylib target/release/libmlp_small.dylib

# Documentation commands
doc:
	@echo "Generating documentation..."
	cargo doc --open --no-deps

doc-private:
	@echo "Generating documentation (including private items)..."
	cargo doc --open --no-deps --document-private-items

# Pre-commit check - runs locally before committing
pre-commit:
	@echo "=== Pre-Commit Checks ==="
	@echo ""
	@echo "1. Formatting code..."
	@cargo fmt || (echo "❌ Format failed!" && exit 1)
	@echo "✅ Code formatted"
	@echo ""
	@echo "2. Running quick lint..."
	@cargo clippy --lib --bins -- -D warnings || (echo "❌ Clippy failed! Fix warnings or run 'cargo clippy --fix'" && exit 1)
	@echo "✅ Lint passed"
	@echo ""
	@echo "3. Running tests..."
	@cargo test || (echo "❌ Tests failed!" && exit 1)
	@echo "✅ Tests passed"
	@echo ""
	@echo "✅ All pre-commit checks passed! Ready to commit."

# CI/CD commands (used by continuous integration)
ci-test:
	@echo "Running CI tests..."
	cargo test --all-targets --all-features

ci-lint:
	@echo "Running CI linting..."
	cargo clippy --all-targets --all-features -- -D warnings
	cargo fmt --check

# All CI checks in one command
ci: ci-lint ci-test
	@echo "All CI checks passed!"
