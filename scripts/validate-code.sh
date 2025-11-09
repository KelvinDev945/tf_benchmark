#!/bin/bash
# Code Validation Script
# Runs the same checks as CI to validate code before pushing

set -e

echo "🔍 Running code validation (same as CI)..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Function to run a check
run_check() {
    local name=$1
    shift
    echo -n "  $name... "
    if "$@" > /tmp/check_output 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        cat /tmp/check_output
        FAILED=1
        return 1
    fi
}

echo "📋 Code Quality Checks:"

# Black
run_check "Black formatting  " black --check --diff src/ tests/ scripts/

# isort
run_check "Import sorting    " isort --check-only --diff src/ tests/ scripts/

# Flake8
run_check "Flake8 linting    " flake8 src/ tests/ scripts/ --max-line-length=100 --extend-ignore=F541,W503

echo ""
echo "🧪 Test Suite:"

# Run tests if they exist
if [ -d "tests" ]; then
    run_check "Unit tests        " pytest tests/ -v -m "not integration and not slow" --tb=short
else
    echo -e "  ${YELLOW}No tests directory found, skipping${NC}"
fi

echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Your code is ready to push! 🚀"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some checks failed.${NC}"
    echo ""
    echo "Please fix the issues above before pushing."
    echo ""
    echo "Quick fixes:"
    echo "  - Run: black src/ tests/ scripts/"
    echo "  - Run: isort src/ tests/ scripts/"
    echo "  - Then re-run: ./scripts/validate-code.sh"
    echo ""
    exit 1
fi
