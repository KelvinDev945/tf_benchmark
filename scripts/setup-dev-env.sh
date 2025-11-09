#!/bin/bash
# Development Environment Setup Script
# Ensures pre-commit hooks are installed and configured correctly

set -e

echo "🔧 Setting up development environment..."

# Install development dependencies if requirements-dev.txt is present
if [ -f "requirements-dev.txt" ]; then
    echo "📦 Installing development dependencies from requirements-dev.txt..."
    pip install -r requirements-dev.txt
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Install commit-msg hook (optional, for commit message validation)
pre-commit install --hook-type commit-msg

# Update hooks to latest version
echo "⬆️  Updating pre-commit hooks..."
pre-commit autoupdate

# Run on all files to ensure everything passes
echo "✅ Running pre-commit on all files (this may take a moment)..."
pre-commit run --all-files || {
    echo ""
    echo "⚠️  Some checks failed. Please fix the issues and run:"
    echo "   pre-commit run --all-files"
    echo ""
    exit 1
}

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Pre-commit hooks will now run automatically on git commit"
echo "  2. To manually run checks: pre-commit run --all-files"
echo "  3. To skip hooks (not recommended): git commit --no-verify"
echo ""
