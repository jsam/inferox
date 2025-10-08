# Release Guide

This document describes how to create a new release of Inferox and publish it to **crates.io** and **docs.rs**.

## 🚀 Quick Start

**Publishing is fully automated via GitHub Actions:**

1. Push a version tag → Workflow publishes to crates.io → docs.rs auto-builds
2. That's it! ✅

## Prerequisites

- You have write access to the repository
- Repository secrets are configured:
  - `CARGO_REGISTRY_TOKEN` - Token from crates.io for publishing
- **Optional but recommended**: Install `git-cliff` for automatic changelog generation:
  ```bash
  cargo install git-cliff
  ```
  If not installed, the release workflow will skip changelog generation.

## Release Methods

There are three ways to trigger a release:

### Method 1: Tag Push (Recommended)

This is the simplest method. Just create and push a version tag:

```bash
# 1. Prepare the release (updates versions + generates changelog)
make release-prepare VERSION=0.1.0

# This will:
# - Update version in all Cargo.toml files
# - Generate CHANGELOG.md with git-cliff
# - Update version references in README.md

# 2. Review the changes
git diff

# 3. Commit the version changes
git add .
git commit -m "chore: release 0.1.0"

# 4. Create and push the tag
git tag 0.1.0
git push origin 0.1.0

# The release workflow will automatically:
# - Run all pre-publish checks
# - Verify version matches tag
# - Publish to crates.io (inferox-core, inferox-candle, inferox-engine)
# - Verify the published packages
```

### Method 2: GitHub Release

Create a release through the GitHub UI:

```bash
# 1. Prepare the release (includes changelog generation)
make release-prepare VERSION=0.1.0

# 2. Commit and push
git add .
git commit -m "chore: release 0.1.0"
git push

# 3. Go to GitHub > Releases > Create a new release
# - Tag: 0.1.0
# - Title: Release 0.1.0
# - Description: Copy from generated CHANGELOG.md
# - Click "Publish release"

# The workflow will run automatically
```

### Method 3: Manual Workflow Dispatch

Trigger the workflow manually from GitHub Actions:

```bash
# 1. Go to Actions > Release > Run workflow
# 2. Select branch
# 3. Enter version (e.g., 0.1.0)
# 4. Choose dry run or actual publish
```

## Pre-Release Checklist

Before creating a release, ensure:

- [ ] All tests pass locally: `make test`
- [ ] Code is formatted: `make format`
- [ ] No clippy warnings: `make lint`
- [ ] Examples compile: `make examples`
- [ ] Model libraries build: `make models`
- [ ] Documentation builds: `make doc`
- [ ] CHANGELOG.md is updated (if exists)
- [ ] Version is correct in all Cargo.toml files
- [ ] All PRs are merged to main

Quick check everything:
```bash
make publish-check
```

## Release Workflow Steps

When you push a tag, the workflow automatically:

### 1. Validation (5-10 minutes)
- ✅ Formats code
- ✅ Runs Clippy
- ✅ Runs all tests on multiple platforms
- ✅ Verifies version matches tag
- ✅ Builds model libraries
- ✅ Builds documentation
- ✅ Builds examples

### 2. Publishing (3-5 minutes)
- ✅ Publishes inferox-core to crates.io
- ✅ Waits for indexing
- ✅ Publishes inferox-candle to crates.io
- ✅ Waits for indexing
- ✅ Publishes inferox-engine to crates.io
- ✅ Creates job summary with install instructions

### 3. Verification (2-3 minutes)
- ✅ Waits for crates.io to index
- ✅ Verifies packages are available
- ✅ Tests installation

### 4. Announcement
- ✅ Creates summary with all links
- ✅ Marks workflow as successful

Total time: ~10-20 minutes

## After Release

Once the workflow completes:

1. **Verify on crates.io**
   - Visit https://crates.io/crates/inferox-core
   - Visit https://crates.io/crates/inferox-candle
   - Visit https://crates.io/crates/inferox-engine
   - Confirm new version is listed

2. **Check documentation (docs.rs auto-builds when you publish to crates.io)**
   - Visit https://docs.rs/inferox-core
   - Visit https://docs.rs/inferox-candle
   - Visit https://docs.rs/inferox-engine
   - Confirm docs are building (may take 5-10 minutes)
   - **No manual action needed** - docs.rs monitors crates.io and builds automatically!

3. **Update dependent projects** (if any)
   ```bash
   cargo update -p inferox-core
   cargo update -p inferox-candle
   cargo update -p inferox-engine
   ```

## Dry Run Testing

To test the release process without publishing:

```bash
# Method 1: Using make
make publish-dry-run

# Method 2: Using GitHub Actions
# Go to Actions > Release > Run workflow
# - Check "Perform dry run only"
# - Click Run workflow
```

This will:
- Run all validation checks
- Show what would be published
- NOT actually publish to crates.io

## Troubleshooting

### Version Mismatch Error

```
❌ Error: Version mismatch!
Cargo.toml version (0.1.0) does not match git tag (0.1.1)
```

**Fix:**
```bash
# Update all Cargo.toml files to match tag
make release-prepare VERSION=0.1.1

# Commit and re-tag
git add .
git commit -m "fix: update version to 0.1.1"
git tag -f 0.1.1  # Force update tag
git push -f origin 0.1.1
```

### Tests Failing

If tests fail during release:

```bash
# Run checks locally
make publish-check

# Fix issues and try again
git add .
git commit -m "fix: resolve test failures"
git push
git tag -f 0.1.1
git push -f origin 0.1.1
```

### Publishing Failed

If publishing to crates.io fails:

1. Check `CARGO_REGISTRY_TOKEN` is valid
2. Verify you have permission to publish
3. Check crates.io status: https://status.crates.io
4. Re-run the workflow from Actions UI

### Dependency Publishing Order

The workflow publishes crates in order:
1. `inferox-core` (no dependencies)
2. `inferox-candle` (depends on inferox-core)
3. `inferox-engine` (depends on inferox-core)

If a dependency fails, subsequent crates won't publish.

## Yanking a Release

If you need to yank a broken release:

```bash
# Yank from crates.io
cargo yank --vers 0.1.0 -p inferox-core
cargo yank --vers 0.1.0 -p inferox-candle
cargo yank --vers 0.1.0 -p inferox-engine

# Or undo the yank
cargo yank --vers 0.1.0 --undo -p inferox-core
cargo yank --vers 0.1.0 --undo -p inferox-candle
cargo yank --vers 0.1.0 --undo -p inferox-engine
```

⚠️ **Note:** Yanking doesn't delete the release, it just prevents new projects from using it.

## Release Checklist Template

Copy this for each release:

```markdown
## Release 0.x.x Checklist

### Pre-Release
- [ ] Update CHANGELOG.md
- [ ] Run `make release-prepare VERSION=0.x.x`
- [ ] Run `make publish-check` locally
- [ ] All tests passing on CI
- [ ] Documentation reviewed

### Release
- [ ] Commit version changes
- [ ] Create and push tag: `git tag 0.x.x && git push origin 0.x.x`
- [ ] Verify workflow started on GitHub Actions
- [ ] Monitor workflow progress (~10-20 min)

### Post-Release
- [ ] Verify on crates.io (all 3 crates)
- [ ] Check docs.rs (all 3 crates)
- [ ] Update dependent projects
- [ ] Announce on social media / Discord / etc. (optional)

### Rollback (if needed)
- [ ] Yank all crates: `cargo yank --vers 0.x.x -p <crate>`
- [ ] Fix issues
- [ ] Release patch version
```

## Version Numbers

Follow semantic versioning (https://semver.org/):

- **Patch** (0.1.1): Bug fixes, no breaking changes
- **Minor** (0.2.0): New features, no breaking changes  
- **Major** (1.0.0): Breaking changes

For pre-1.0 releases, minor versions may contain breaking changes.

## Crate Publishing Order

The crates must be published in dependency order:

1. **inferox-core** - Core traits and types (no dependencies)
2. **inferox-candle** - Candle backend (depends on inferox-core)
3. **inferox-engine** - Engine runtime (depends on inferox-core)

The release workflow handles this automatically with wait periods between publishes.

## Getting Help

If you encounter issues:

1. Check workflow logs in GitHub Actions
2. Review this guide
3. Check `.github/workflows/release.yml` for details
4. Open an issue if you find a bug in the release process

## Security

- Never commit `CARGO_REGISTRY_TOKEN` to the repository
- Always use GitHub Secrets for sensitive tokens
- Review the diff before pushing tags
- Be cautious with force-pushing tags

---

## Cheat Sheet

```bash
# 1. Prepare release (updates versions + generates CHANGELOG)
make release-prepare VERSION=0.1.0

# 2. Create PR (standard pr-checks.yml validates)
git checkout -b release/0.1.0
git add .
git commit -m "chore: release 0.1.0"
git push -u origin release/0.1.0

# 3. Merge PR to main

# 4. Tag release (triggers release.yml workflow)
git checkout main
git pull
git tag 0.1.0
git push origin 0.1.0  # 🚀 Automatically publishes to crates.io
```

## First Release (0.1.0)

For the initial release:

```bash
# 1. Prepare release (includes git-cliff changelog generation)
make release-prepare VERSION=0.1.0

# 2. Run full validation
make publish-check

# 3. Create release
git add .
git commit -m "chore: release 0.1.0"
git push
git tag 0.1.0
git push origin 0.1.0

# 4. Monitor the release workflow
# Visit: https://github.com/jsam/inferox/actions/workflows/release.yml
```
