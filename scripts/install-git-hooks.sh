#!/bin/bash
# Install git hooks for Inferox development
# Run this script once to set up pre-commit hooks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

echo "Installing git hooks for Inferox..."
echo ""

# Check if we're in a git repository
if [ ! -d "$REPO_ROOT/.git" ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Create pre-commit hook
echo "📝 Creating pre-commit hook..."
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Inferox Pre-Commit Hook
# Runs formatting check and quick lints before allowing commit

set -e

echo "🔍 Running pre-commit checks..."
echo ""

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Not in repository root"
    exit 1
fi

# 1. Format Check
echo "1/3 Checking code formatting..."
if ! cargo fmt --check --quiet; then
    echo ""
    echo "❌ Code formatting check failed!"
    echo ""
    echo "Your code is not properly formatted."
    echo "Run 'cargo fmt' to fix formatting automatically, then commit again."
    echo ""
    exit 1
fi
echo "   ✅ Formatting OK"
echo ""

# 2. Quick Clippy Check (lib + bins only, faster than full check)
echo "2/3 Running quick lint check..."
CLIPPY_OUTPUT=$(cargo clippy --lib --bins --quiet -- -D warnings 2>&1)
CLIPPY_EXIT=$?
if [ $CLIPPY_EXIT -ne 0 ]; then
    echo ""
    echo "$CLIPPY_OUTPUT"
    echo ""
    echo "❌ Clippy check failed!"
    echo ""
    echo "Fix the warnings above or run 'cargo clippy --fix' to auto-fix."
    echo ""
    exit 1
fi
echo "   ✅ Lints OK"
echo ""

# 3. Quick compile check
echo "3/3 Checking code compiles..."
if ! cargo check --quiet 2>&1 | tail -5; then
    echo ""
    echo "❌ Compilation check failed!"
    echo ""
    echo "Fix the compilation errors above before committing."
    echo ""
    exit 1
fi
echo "   ✅ Compiles OK"
echo ""

echo "✅ All pre-commit checks passed!"
echo ""
echo "💡 Tip: Run 'make test' before pushing to run full test suite."
echo ""

exit 0
EOF

# Make it executable
chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed at: $HOOKS_DIR/pre-commit"
echo ""
echo "The hook will automatically run on 'git commit' and check:"
echo "  • Code formatting (cargo fmt --check)"
echo "  • Quick lints (cargo clippy)"
echo "  • Compilation (cargo check)"
echo ""
echo "To skip the hook for a specific commit, use: git commit --no-verify"
echo ""
