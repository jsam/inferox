# Development Scripts

This directory contains helpful scripts for Inferox development.

## Git Hooks

### Installing Git Hooks

Run the install script to set up pre-commit hooks:

```bash
./scripts/install-git-hooks.sh
```

This will install a pre-commit hook that automatically runs before each commit.

### What the Pre-Commit Hook Does

The pre-commit hook runs three quick checks:

1. **Code Formatting** (`cargo fmt --check`)
   - Ensures all code follows Rust formatting standards
   - If it fails, run `cargo fmt` to auto-fix

2. **Quick Lints** (`cargo clippy --lib --bins`)
   - Runs Clippy on library and binary code (faster than full check)
   - Catches common mistakes and style issues
   - If it fails, run `cargo clippy --fix` to auto-fix many issues

3. **Compilation Check** (`cargo check`)
   - Verifies the code compiles
   - Catches syntax errors and type issues

### Bypassing the Hook

If you need to commit without running the hook (not recommended), use:

```bash
git commit --no-verify
```

### Benefits

- Catches formatting and lint issues before they reach CI
- Faster feedback loop (local check vs waiting for CI)
- Keeps the codebase clean and consistent
- Reduces back-and-forth in pull request reviews

### Manual Alternative

If you prefer not to use git hooks, you can run the same checks manually:

```bash
make pre-commit
```

This runs the same checks plus the full test suite.
