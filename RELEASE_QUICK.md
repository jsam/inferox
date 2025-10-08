# Quick Release Guide

## TL;DR - How to Release

```bash
# 1. Bump version
vim Cargo.toml  # Update version = "0.2.0"

# 2. Validate
make publish-check

# 3. Commit
git add Cargo.toml
git commit -m "chore: bump version to 0.2.0"
git push

# 4. Tag and push (this triggers everything)
git tag 0.2.0
git push origin 0.2.0
```

**That's it!** The GitHub Actions workflow will:
- ✅ Run all validation checks
- 📦 Publish to crates.io (inferox-core → inferox-candle → inferox-engine)
- 📚 Trigger docs.rs build (automatic when published to crates.io)
- ✔️ Verify packages are available

## What Happens Automatically

### When you push a tag (e.g., `0.2.0`):

1. **GitHub Actions starts** (`.github/workflows/release.yml`)
2. **Validates** (format, lint, tests, docs)
3. **Publishes to crates.io** in sequence:
   - `inferox-core` (no deps)
   - Wait 60s for propagation
   - `inferox-candle` (depends on core)
   - Wait 60s for propagation
   - `inferox-engine` (depends on core)
4. **docs.rs automatically detects** new versions on crates.io
5. **docs.rs builds** documentation for all 3 crates
6. **Workflow verifies** packages are available
7. **Done!** ✅

## docs.rs - No Manual Action Required

**docs.rs monitors crates.io and builds automatically!**

When you publish to crates.io, docs.rs:
1. Detects new version within ~1 minute
2. Queues a build
3. Builds documentation
4. Publishes to https://docs.rs/crate-name/version

**You don't need to do anything** - just publish to crates.io and wait 5-10 minutes.

## Checking the Release

After ~5 minutes:

```bash
# Crates.io
open https://crates.io/crates/inferox-core
open https://crates.io/crates/inferox-candle
open https://crates.io/crates/inferox-engine

# docs.rs (wait 5-10 min for build)
open https://docs.rs/inferox-core
open https://docs.rs/inferox-candle
open https://docs.rs/inferox-engine
```

## Prerequisites (One-Time Setup)

### 1. Add CARGO_REGISTRY_TOKEN to GitHub Secrets

1. Get token: https://crates.io/me
2. GitHub repo → Settings → Secrets → Actions
3. New secret: `CARGO_REGISTRY_TOKEN` = your token

### 2. Be an owner on crates.io

You must be an owner of:
- `inferox-core`
- `inferox-candle`
- `inferox-engine`

## Tag Format

Use **plain version numbers** (no 'v' prefix):

✅ `0.1.0`
✅ `0.2.0-beta.1`
✅ `1.0.0`

❌ `v0.1.0` (don't use 'v')

## Troubleshooting

### Version mismatch error

**Error:** `Cargo.toml version (0.1.0) does not match git tag (0.2.0)`

**Fix:**
```bash
vim Cargo.toml  # Update version
git add Cargo.toml
git commit -m "fix: version"
git tag -f 0.2.0  # Force update tag
git push origin 0.2.0 --force
```

### Workflow failed

1. Check logs: Actions → Release workflow
2. Fix issue
3. Re-tag: `git tag -f 0.2.0 && git push origin 0.2.0 --force`

### docs.rs not building

1. Wait 10-15 minutes (builds can take time)
2. Check queue: https://docs.rs/releases/queue
3. Check build logs: https://docs.rs/crate/inferox-core/latest/builds

## Dry Run (Test Without Publishing)

```bash
# Local
make publish-dry-run

# Or GitHub Actions
# Actions → Release → Run workflow → Check "dry run"
```

## Full Documentation

See [RELEASE.md](./RELEASE.md) for complete details.

## Summary

**The entire release process is automated:**

```
git tag 0.2.0 → GitHub Actions → crates.io → docs.rs
            ↓
        Everything auto-publishes
```

No manual `cargo publish` needed. No manual docs.rs actions needed.
