#!/bin/bash
# Publish script for Inferox workspace
# Publishes crates in dependency order to crates.io

set -e

echo "=================================="
echo "Inferox Publishing Script"
echo "=================================="
echo ""

# Check if we're doing a dry run
DRY_RUN=""
if [ "$1" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "🔍 Running in DRY RUN mode (no actual publish)"
    echo ""
fi

# Check git status
if [ -z "$DRY_RUN" ]; then
    if ! git diff-index --quiet HEAD --; then
        echo "❌ Error: Working directory has uncommitted changes"
        echo "Please commit or stash changes before publishing"
        exit 1
    fi
fi

# Function to publish a crate
publish_crate() {
    local crate=$1
    echo "📦 Publishing $crate..."
    
    if [ -n "$DRY_RUN" ]; then
        cargo publish -p "$crate" --dry-run
    else
        cargo publish -p "$crate"
        # Wait for crates.io to propagate (usually takes 30-60 seconds)
        if [ "$crate" != "inferox-engine" ]; then
            echo "⏳ Waiting 60s for crates.io to propagate..."
            sleep 60
        fi
    fi
    
    echo "✅ $crate published successfully"
    echo ""
}

# Publish in dependency order
echo "Publishing crates in dependency order:"
echo "  1. inferox-core (no workspace dependencies)"
echo "  2. inferox-candle (depends on inferox-core)"
echo "  3. inferox-engine (depends on inferox-core)"
echo ""

if [ -z "$DRY_RUN" ]; then
    read -p "Continue with publish? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    echo ""
fi

# Publish each crate
publish_crate "inferox-core"
publish_crate "inferox-candle"
publish_crate "inferox-engine"

echo "=================================="
echo "✅ All crates published successfully!"
echo "=================================="
echo ""
echo "Published versions:"
VERSION=$(cargo metadata --format-version 1 --no-deps | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "  - inferox-core v$VERSION"
echo "  - inferox-candle v$VERSION"
echo "  - inferox-engine v$VERSION"
echo ""
echo "🎉 Release complete!"
