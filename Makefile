# Inferox Makefile
# Provides convenient commands for testing, building, and development

.PHONY: help build build-release test test-quick lint lint-quick pre-commit clean doc examples models run-example release-prepare publish-check publish-dry-run coverage coverage-check

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
	@echo ""
	@echo "Release:"
	@echo "  release-prepare VERSION=x.y.z - Prepare release by updating versions"
	@echo "  publish-check   - Run all pre-publish validation checks"
	@echo "  publish-dry-run - Test publish without actually publishing"
	@echo ""
	@echo "Coverage:"
	@echo "  coverage        - Generate coverage report"
	@echo "  coverage-check  - Check if coverage meets threshold"

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

# Release commands
release-prepare:
	@echo "Preparing release $(VERSION)..."
	@if [ -z "$(VERSION)" ]; then \
		echo "❌ Error: VERSION not specified"; \
		echo "Usage: make release-prepare VERSION=0.1.0"; \
		exit 1; \
	fi
	@echo ""
	@echo "1️⃣  Updating version to $(VERSION) in all Cargo.toml files..."
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' crates/inferox-core/Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' crates/inferox-candle/Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' crates/inferox-engine/Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' examples/mlp/Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' examples/mlp/models/classifier/Cargo.toml
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' examples/mlp/models/small/Cargo.toml
	@find . -name "*.bak" -delete
	@echo "   ✅ Version updated to $(VERSION)"
	@echo ""
	@echo "2️⃣  Generating changelog with git-cliff..."
	@if command -v git-cliff >/dev/null 2>&1; then \
		git cliff --tag $(VERSION) -o CHANGELOG.md && \
		echo "   ✅ Changelog generated"; \
	else \
		echo "   ⚠️  git-cliff not installed. Install with: cargo install git-cliff"; \
		echo "   Skipping changelog generation..."; \
	fi
	@echo ""
	@echo "3️⃣  Updating README version references..."
	@sed -i.bak 's/inferox-core = "[^"]*"/inferox-core = "$(VERSION)"/g' README.md
	@sed -i.bak 's/inferox-candle = "[^"]*"/inferox-candle = "$(VERSION)"/g' README.md
	@sed -i.bak 's/inferox-engine = "[^"]*"/inferox-engine = "$(VERSION)"/g' README.md
	@find . -name "*.bak" -delete
	@echo "   ✅ README updated"
	@echo ""
	@echo "✅ Release preparation complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "1. Review changes: git diff"
	@echo "2. Run checks: make publish-check"
	@echo "3. Commit: git add . && git commit -m 'chore: release $(VERSION)'"
	@echo "4. Tag: git tag $(VERSION)"
	@echo "5. Push: git push origin $(VERSION)"

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
	@echo "6. Checking package..."
	@cd crates/inferox-core && cargo package --allow-dirty > /dev/null 2>&1 || (echo "❌ inferox-core package check failed!" && exit 1)
	@cd crates/inferox-candle && cargo package --allow-dirty > /dev/null 2>&1 || (echo "❌ inferox-candle package check failed!" && exit 1)
	@cd crates/inferox-engine && cargo package --allow-dirty > /dev/null 2>&1 || (echo "❌ inferox-engine package check failed!" && exit 1)
	@echo "✅ Package checks passed"
	@echo ""
	@echo "✅ All publish checks passed! Ready to release."

publish-dry-run:
	@echo "Running dry-run publish..."
	@echo ""
	@echo "inferox-core:"
	@cd crates/inferox-core && cargo publish --dry-run
	@echo ""
	@echo "inferox-candle:"
	@cd crates/inferox-candle && cargo publish --dry-run
	@echo ""
	@echo "inferox-engine:"
	@cd crates/inferox-engine && cargo publish --dry-run
	@echo ""
	@echo "✅ Dry-run completed successfully!"

# Coverage commands
coverage:
	@echo "Generating coverage report..."
	@mkdir -p target/coverage
	@cargo tarpaulin \
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
		--out Json \
		--output-dir target/coverage \
		--exclude-files examples/* \
		--timeout 300 > /dev/null 2>&1
	@if [ -f target/coverage/tarpaulin-report.json ]; then \
		COVERAGE=$$(jq -r '.coverage' target/coverage/tarpaulin-report.json 2>/dev/null || echo "0"); \
		THRESHOLD=65.0; \
		echo "Coverage: $$COVERAGE%"; \
		echo "Threshold: $$THRESHOLD%"; \
		if [ $$(echo "$$COVERAGE >= $$THRESHOLD" | bc -l) -eq 1 ]; then \
			echo "✅ Coverage meets threshold!"; \
		else \
			echo "❌ Coverage below threshold!"; \
			exit 1; \
		fi; \
	else \
		echo "❌ Coverage report not found!"; \
		exit 1; \
	fi
