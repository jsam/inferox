# Publishing Guide

This document explains how to publish Inferox crates to crates.io.

## Workspace Structure

Inferox is a multi-crate workspace with the following **publishable** crates:

```
crates/
├── inferox-core     - Core traits and types (no workspace deps)
├── inferox-candle   - Candle backend (depends on inferox-core)
└── inferox-engine   - Inference engine (depends on inferox-core)
```

The `examples/` directory contains non-publishable demo crates marked with `publish = false`.

## Publishing Order

Crates **must** be published in dependency order:

1. **inferox-core** (first - no workspace dependencies)
2. **inferox-candle** (after inferox-core is on crates.io)
3. **inferox-engine** (after inferox-core is on crates.io)

## Quick Start

### 1. Pre-publish Checks

```bash
make publish-check
```

This runs:
- Format check (`cargo fmt --check`)
- Clippy linting (`cargo clippy`)
- All tests (`cargo test`)
- Model builds
- Documentation build
- Package metadata validation

### 2. Dry Run

```bash
make publish-dry-run
```

This simulates publishing without actually uploading to crates.io.

**Note:** The dry-run will fail for `inferox-candle` and `inferox-engine` if `inferox-core` hasn't been published yet, because cargo validates dependencies against crates.io. This is expected.

### 3. Publish

```bash
make publish
```

This runs the automated publish script which:
- Checks for uncommitted changes
- Publishes crates in dependency order
- Waits 60s between publishes for crates.io propagation
- Shows confirmation before proceeding

## Manual Publishing

If you need to publish manually:

```bash
# 1. Publish core first
cargo publish -p inferox-core

# 2. Wait for propagation (30-60 seconds)
sleep 60

# 3. Publish candle
cargo publish -p inferox-candle

# 4. Wait for propagation
sleep 60

# 5. Publish engine
cargo publish -p inferox-engine
```

## Version Management

All crates share the same version defined in the workspace `Cargo.toml`:

```toml
[workspace.package]
version = "0.1.0"
```

To bump the version:

1. Update `version` in `Cargo.toml` (workspace root)
2. Commit the change
3. Create a git tag: `git tag v0.2.0`
4. Run `make publish`

## Troubleshooting

### "no matching package named `inferox-core` found"

This error occurs during dry-run or publish of `inferox-candle`/`inferox-engine` if `inferox-core` isn't on crates.io yet.

**Solution:** Publish `inferox-core` first, wait 60s, then publish the others.

### "uncommitted changes" error

The publish script prevents publishing with uncommitted changes.

**Solution:** Commit or stash your changes before publishing.

### Package verification fails

If `cargo package` fails for a crate:

```bash
# Check what files will be included
cargo package --list -p <crate-name>

# Package locally to inspect
cargo package -p <crate-name> --allow-dirty
```

## Crates.io Links

After publishing, crates will be available at:

- https://crates.io/crates/inferox-core
- https://crates.io/crates/inferox-candle
- https://crates.io/crates/inferox-engine

## CI/CD Integration

For automated publishing in CI:

```yaml
# .github/workflows/publish.yml
- name: Publish to crates.io
  env:
    CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_TOKEN }}
  run: |
    cargo publish -p inferox-core
    sleep 60
    cargo publish -p inferox-candle
    sleep 60
    cargo publish -p inferox-engine
```

## Yanking a Release

If you need to yank a bad release:

```bash
cargo yank --vers 0.1.0 inferox-core
cargo yank --vers 0.1.0 inferox-candle
cargo yank --vers 0.1.0 inferox-engine
```

## Best Practices

1. **Always run `make publish-check` first**
2. **Test locally before publishing** - run examples with the packaged code
3. **Use semantic versioning** - follow [SemVer](https://semver.org/)
4. **Update CHANGELOG.md** before releasing
5. **Tag releases** in git: `git tag v0.2.0`
6. **Never force-push** after publishing a version
7. **Test the published crates** immediately after release:
   ```bash
   cargo new test-inferox
   cd test-inferox
   cargo add inferox-core inferox-candle inferox-engine
   cargo build
   ```
