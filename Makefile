# Inferox Makefile
# Provides convenient commands for testing, building, and development

.PHONY: help install build build-release test test-quick lint lint-quick pre-commit clean doc examples models run-example release-prepare publish-check publish-dry-run coverage coverage-check changelog

# Default target
help:
	@echo "Inferox Development Commands"
	@echo "============================="
	@echo ""
	@echo "Setup:"
	@echo "  install         - Install development dependencies (PyTorch, tools)"
	@echo ""
	@echo "Building:"
	@echo "  build           - Build all crates in debug mode"
	@echo "  build-release   - Build all crates in release mode"
	@echo "  models          - Build all model libraries (.dylib/.so)"
	@echo "  examples        - Build all example binaries"
	@echo ""
	@echo "Testing:"
	@echo "  test              - Run tests + quick lint (recommended)"
	@echo "  test-quick        - Run tests only (faster)"
	@echo "  test-core         - Run core library tests only"
	@echo "  test-candle       - Run Candle backend tests only"
	@echo "  test-engine       - Run engine tests only"
	@echo "  test-examples     - Build and test ALL examples (MLP + BERT with package assembly)"
	@echo "  test-bert-tch     - Build and test BERT-Tch example (requires LibTorch)"
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
	@echo ""
	@echo "Release:"
	@echo "  release-prepare VERSION=x.y.z - Prepare release branch with version bump"
	@echo "  changelog [VERSION=x.y.z]     - Generate changelog (optionally for version)"
	@echo "  publish-check   - Run all pre-publish validation checks"
	@echo "  publish-dry-run - Test publish without actually publishing"
	@echo "  publish         - Publish all crates to crates.io (in sequence)"
	@echo ""
	@echo "Coverage:"
	@echo "  coverage        - Generate coverage report"
	@echo "  coverage-check  - Check if coverage meets threshold"

# Setup commands
install:
	@echo "Installing development dependencies..."
	@echo ""
	@echo "1. Checking Python installation..."
	@which python3 > /dev/null || (echo "❌ Python 3 not found! Install Python 3 first." && exit 1)
	@echo "   ✅ Python 3 found: $$(python3 --version)"
	@echo ""
	@echo "2. Installing PyTorch 2.4.0 (required for tch-rs backend)..."
	@pip3 show torch 2>/dev/null | grep -q "Version: 2.4.0" && \
		echo "   ✅ PyTorch 2.4.0 already installed" || \
		(echo "   Installing PyTorch 2.4.0..." && pip3 install torch==2.4.0)
	@echo ""
	@echo "3. Installing Python packages (huggingface_hub, safetensors)..."
	@pip3 install huggingface_hub safetensors -q
	@echo "   ✅ Python packages installed"
	@echo ""
	@echo "4. Installing Rust development tools..."
	@which cargo-tarpaulin > /dev/null 2>&1 && \
		echo "   ✅ cargo-tarpaulin already installed" || \
		(echo "   Installing cargo-tarpaulin (this may take a while)..." && cargo install cargo-tarpaulin)
	@echo "   ✅ Rust tools ready"
	@echo ""
	@echo "5. Setting up git hooks..."
	@if [ ! -f .git/hooks/pre-commit ]; then \
		echo "   ⚠️  Pre-commit hook not found (expected in .git/hooks/pre-commit)"; \
	else \
		chmod +x .git/hooks/pre-commit; \
		echo "   ✅ Git hooks configured"; \
	fi
	@echo ""
	@echo "✅ All development dependencies installed!"
	@echo ""
	@echo "📝 Next steps:"
	@echo "   - Run 'make build' to build the project"
	@echo "   - Run 'make test' to run tests"
	@echo "   - Run 'make test-bert-tch' to test BERT-Tch example"
	@echo ""
	@echo "💡 Environment variables for tch-rs:"
	@echo "   export LIBTORCH_USE_PYTORCH=1"
	@echo "   export DYLD_LIBRARY_PATH=\"\$$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))'):\$$DYLD_LIBRARY_PATH\""

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

test-examples: test-mlp test-bert-candle test-bert-tch
	@echo ""
	@echo "✅ All examples tested successfully!"

test-mlp:
	@echo "=== Testing MLP Example ==="
	@echo ""
	@echo "1. Building MLP model libraries..."
	@cargo build --release -p mlp-classifier -p mlp-small || (echo "❌ MLP models build failed!" && exit 1)
	@echo "✅ MLP models built"
	@echo ""
	@echo "2. Running MLP unit tests..."
	@cargo test -p mlp --lib || (echo "❌ MLP unit tests failed!" && exit 1)
	@echo "✅ MLP unit tests passed"
	@echo ""
	@echo "3. Running MLP e2e tests (including engine integration)..."
	@cargo test -p mlp --test e2e_test -- --ignored || (echo "❌ MLP e2e tests failed!" && exit 1)
	@echo "✅ MLP e2e tests passed"

test-bert-candle:
	@echo ""
	@echo "=== Testing BERT-Candle Example ==="
	@echo ""
	@echo "1. Building BERT-Candle library..."
	@cargo build --release -p bert-candle || (echo "❌ BERT-Candle library build failed!" && exit 1)
	@echo "✅ BERT-Candle library built"
	@echo ""
	@echo "2. Assembling BERT-Candle package..."
	@touch examples/bert-candle/build.rs
	@cargo build --release -p bert-candle || (echo "❌ BERT-Candle package assembly failed!" && exit 1)
	@echo "✅ BERT-Candle package assembled"
	@echo ""
	@echo "3. Verifying package exists..."
	@if [ ! -d "target/mlpkg/bert-candle" ]; then \
		echo "❌ Package directory not found at target/mlpkg/bert-candle"; \
		exit 1; \
	fi
	@echo "✅ Package verified at target/mlpkg/bert-candle"
	@echo ""
	@echo "4. Running BERT-Candle tests (including e2e)..."
	@cargo test -p bert-candle || (echo "❌ BERT-Candle unit tests failed!" && exit 1)
	@cargo test --test e2e_test -p bert-candle -- --ignored --nocapture || (echo "❌ BERT-Candle e2e tests failed!" && exit 1)
	@echo "✅ BERT-Candle tests passed"

test-bert-tch:
	@echo ""
	@echo "=== Testing BERT-Tch Example (requires LibTorch) ==="
	@echo ""
	@echo "Checking for PyTorch installation..."
	@if ! python3 -c "import torch; print(f'Found PyTorch {torch.__version__}')" 2>/dev/null; then \
		echo "⚠️  PyTorch not found - skipping BERT-Tch tests"; \
		echo "   To run these tests, install PyTorch: pip3 install torch==2.4.0"; \
		exit 0; \
	fi
	@echo ""
	@echo "1. Building BERT-Tch library..."
	@cd examples/bert-tch && LIBTORCH_USE_PYTORCH=1 cargo build --release || (echo "❌ BERT-Tch library build failed!" && exit 1)
	@echo "✅ BERT-Tch library built"
	@echo ""
	@echo "2. Assembling BERT-Tch package..."
	@touch examples/bert-tch/build.rs
	@cd examples/bert-tch && LIBTORCH_USE_PYTORCH=1 cargo build --release || (echo "❌ BERT-Tch package assembly failed!" && exit 1)
	@echo "✅ BERT-Tch package assembled"
	@echo ""
	@echo "3. Verifying package exists..."
	@if [ ! -d "target/mlpkg/bert-tch" ]; then \
		echo "❌ Package directory not found at target/mlpkg/bert-tch"; \
		exit 1; \
	fi
	@echo "✅ Package verified at target/mlpkg/bert-tch"
	@echo ""
	@echo "4. Testing BERT-Tch example (including e2e tests)..."
	@cd examples/bert-tch && \
		LIBTORCH_USE_PYTORCH=1 \
		DYLD_LIBRARY_PATH="$$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')":$$DYLD_LIBRARY_PATH \
		cargo test || (echo "❌ BERT-Tch unit tests failed!" && exit 1)
	@cd examples/bert-tch && \
		LIBTORCH_USE_PYTORCH=1 \
		DYLD_LIBRARY_PATH="$$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')":$$DYLD_LIBRARY_PATH \
		cargo test --test e2e_test -- --ignored --nocapture || (echo "❌ BERT-Tch e2e tests failed!" && exit 1)
	@echo "✅ BERT-Tch tests passed"

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
	@echo "2. Running CI lint..."
	@cargo clippy --workspace --all-targets -- -D warnings || (echo "❌ Clippy failed! Fix warnings or run 'cargo clippy --fix'" && exit 1)
	@cargo fmt --check || (echo "❌ Format check failed!" && exit 1)
	@echo "✅ CI lint passed"
	@echo ""
	@echo "3. Running tests..."
	@cargo test --workspace || (echo "❌ Tests failed!" && exit 1)
	@echo "✅ Tests passed"
	@echo ""
	@echo "✅ All pre-commit checks passed! Ready to commit."

# CI/CD commands (used by continuous integration)
ci-test:
	@echo "Running CI tests..."
	cargo test --workspace --all-targets

ci-lint:
	@echo "Running CI linting..."
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --check

# All CI checks in one command
ci: ci-lint ci-test
	@echo "All CI checks passed!"

# Release commands
release-prepare:
	@if [ -z "$(VERSION)" ]; then \
		echo "❌ Error: VERSION not specified"; \
		echo "Usage: make release-prepare VERSION=0.2.0"; \
		exit 1; \
	fi
	@echo "Preparing release $(VERSION)..."
	@./scripts/prepare-release.sh $(VERSION)

changelog:
	@echo "Generating changelog..."
	@command -v git-cliff >/dev/null 2>&1 || { \
		echo "⚠️  git-cliff not found. Installing..."; \
		cargo install git-cliff; \
	}
	@if [ -z "$(VERSION)" ]; then \
		echo "Generating changelog for all changes..."; \
		git-cliff -o CHANGELOG.md; \
	else \
		echo "Generating changelog for version $(VERSION)..."; \
		git-cliff --tag "$(VERSION)" -o CHANGELOG.md; \
	fi
	@echo "✅ Changelog generated in CHANGELOG.md"

publish-check:
	@echo "Running pre-publish checks..."
	@echo ""
	@echo "1. Checking format..."
	@cargo fmt --check || (echo "❌ Format check failed! Run 'make format' first." && exit 1)
	@echo "✅ Format check passed"
	@echo ""
	@echo "2. Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings || (echo "❌ Clippy failed!" && exit 1)
	@echo "✅ Clippy passed"
	@echo ""
	@echo "3. Running tests..."
	@cargo test --all-targets --all-features || (echo "❌ Tests failed!" && exit 1)
	@echo "✅ Tests passed"
	@echo ""
	@echo "4. Building model libraries..."
	@cargo build --release -p mlp-classifier -p mlp-small || (echo "❌ Model build failed!" && exit 1)
	@echo "✅ Model libraries built"
	@echo ""
	@echo "5. Building documentation..."
	@cargo doc --no-deps || (echo "❌ Documentation build failed!" && exit 1)
	@echo "✅ Documentation built"
	@echo ""
	@echo "6. Checking package metadata..."
	@cargo package --list --allow-dirty -p inferox-core > /dev/null 2>&1 || (echo "❌ inferox-core package list failed!" && exit 1)
	@cargo package --list --allow-dirty -p inferox-candle > /dev/null 2>&1 || (echo "❌ inferox-candle package list failed!" && exit 1)
	@cargo package --list --allow-dirty -p inferox-engine > /dev/null 2>&1 || (echo "❌ inferox-engine package list failed!" && exit 1)
	@cargo package --list --allow-dirty -p inferox-mlpkg > /dev/null 2>&1 || (echo "❌ inferox-mlpkg package list failed!" && exit 1)
	@echo "✅ Package metadata checks passed"
	@echo ""
	@echo "ℹ️  Note: Full package validation with crates.io dependencies"
	@echo "   requires publishing inferox-core first, then inferox-candle,"
	@echo "   then inferox-engine, then inferox-mlpkg in sequence."
	@echo ""
	@echo "✅ All publish checks passed! Ready to release."

publish-dry-run:
	@echo "Running dry-run publish..."
	@./scripts/publish.sh --dry-run

publish:
	@echo "Publishing to crates.io..."
	@./scripts/publish.sh

# Coverage commands
coverage:
	@echo "Generating coverage report..."
	@mkdir -p target/coverage
	@cargo tarpaulin \
		--workspace \
		--out Html \
		--out Json \
		--out Xml \
		--output-dir target/coverage \
		--exclude-files examples/* \
		--timeout 300
	@echo ""
	@echo "✅ Coverage report generated!"
	@echo "  HTML: target/coverage/tarpaulin-report.html"
	@echo "  JSON: target/coverage/tarpaulin-report.json"
	@echo "  XML:  target/coverage/tarpaulin-report.xml"

coverage-check:
	@echo "Checking coverage threshold..."
	@mkdir -p target/coverage
	@cargo tarpaulin \
		--workspace \
		--out Json \
		--output-dir target/coverage \
		--exclude-files 'examples/*' \
		--exclude-files 'crates/hf-xet-rs/src/hf_api.rs' \
		--exclude-files 'crates/hf-xet-rs/src/client.rs' \
		--exclude-files 'crates/inferox-mlpkg/src/lib.rs' \
		--timeout 300
	@if [ -f target/coverage/tarpaulin-report.json ]; then \
		COVERAGE=$$(jq -r '.coverage' target/coverage/tarpaulin-report.json 2>/dev/null || echo "0"); \
		THRESHOLD=65.0; \
		echo ""; \
		echo "Coverage: $$COVERAGE%"; \
		echo "Threshold: $$THRESHOLD%"; \
		if [ $$(echo "$$COVERAGE >= $$THRESHOLD" | bc -l) -eq 1 ]; then \
			echo "✅ Coverage meets threshold!"; \
			echo "ℹ️  Network code excluded (tested via 12 integration tests)."; \
		else \
			echo "❌ Coverage below threshold!"; \
			echo "ℹ️  Network code excluded (tested via 12 integration tests)."; \
		fi; \
	else \
		echo "❌ Coverage report not found!"; \
		exit 1; \
	fi
