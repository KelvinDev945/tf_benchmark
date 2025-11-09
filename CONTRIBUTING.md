# Contributing to TensorFlow Benchmark

Thank you for contributing! This guide will help you set up your development environment and ensure your code meets our quality standards.

## Quick Start

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/KelvinDev945/tf_benchmark.git
cd tf_benchmark

# Run the setup script (installs pre-commit hooks)
./scripts/setup-dev-env.sh
```

This script will:
- Install pre-commit framework
- Configure git hooks to run automatically
- Update hooks to latest versions
- Validate that all checks pass

### 2. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Before committing, validate your code
./scripts/validate-code.sh

# Commit (pre-commit hooks will run automatically)
git commit -m "Your commit message"

# If hooks fail, fix the issues and try again
black src/ tests/ scripts/
isort src/ tests/ scripts/
git add .
git commit -m "Your commit message"

# Push your changes
git push origin feature/your-feature-name
```

## Code Quality Standards

All code must pass these checks before being merged:

### 1. **Black** - Code Formatting
```bash
# Check formatting
black --check src/ tests/ scripts/

# Auto-fix formatting
black src/ tests/ scripts/
```

Configuration:
- Line length: 100 characters
- Python version: 3.11+

### 2. **isort** - Import Sorting
```bash
# Check import order
isort --check-only src/ tests/ scripts/

# Auto-fix imports
isort src/ tests/ scripts/
```

Configuration:
- Profile: black
- Line length: 100 characters

### 3. **Flake8** - Linting
```bash
# Run linter
flake8 src/ tests/ scripts/ --max-line-length=100
```

Configuration:
- Max line length: 100
- Ignored: F541 (f-string without placeholders), W503 (line break before operator)

### 4. **pytest** - Tests
```bash
# Run unit tests
pytest tests/ -v -m "not integration and not slow"

# Run all tests
pytest tests/ -v
```

## Pre-commit Hooks

Pre-commit hooks ensure code quality **before** committing:

### How They Work

1. **Automatic**: Hooks run on every `git commit`
2. **Fast**: Only check changed files
3. **Consistent**: Same checks as CI/CD

### What Hooks Check

- ✅ Trailing whitespace
- ✅ End-of-file newlines
- ✅ YAML/TOML/JSON syntax
- ✅ Black formatting
- ✅ Import sorting (isort)
- ✅ Flake8 linting

### Bypass Hooks (Not Recommended)

```bash
# Skip hooks for emergency commits only
git commit --no-verify -m "Emergency fix"
```

⚠️ **Warning**: Bypassing hooks will cause CI to fail!

## CI/CD Pipeline

Our CI runs the same checks as pre-commit hooks:

### Workflow

1. **Code Quality** - Black, isort, Flake8
2. **Tests** - pytest with coverage
3. **Documentation** - API docs generation
4. **Docker** - Build container images

### Debugging CI Failures

If CI fails but local checks pass:

```bash
# Run the exact same checks as CI
./scripts/validate-code.sh

# Or manually:
black --check --diff src/ tests/ scripts/
isort --check-only --diff src/ tests/ scripts/
flake8 src/ tests/ scripts/ --max-line-length=100
pytest tests/ -v
```

## Common Issues

### 1. Pre-commit hooks not running

```bash
# Reinstall hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

### 2. Formatting issues

```bash
# Auto-fix all formatting
black src/ tests/ scripts/
isort src/ tests/ scripts/
```

### 3. Import order errors

```bash
# isort will automatically organize imports
isort src/ tests/ scripts/
```

### 4. Pre-commit failing on all files

```bash
# Run on specific files only
pre-commit run --files src/your_file.py

# Skip certain hooks
SKIP=flake8 git commit -m "..."
```

## Best Practices

### Commit Messages

Follow the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Examples:
```
feat(benchmark): Add ONNX conversion comparison
fix(dataset): Handle missing datasets library gracefully
docs(readme): Update installation instructions
test(models): Add unit tests for ModelLoader
```

### Code Style

- Use type hints where possible
- Write docstrings for public functions/classes
- Keep functions small and focused
- Add tests for new features

### Testing

- Write unit tests for new code
- Ensure tests pass locally before pushing
- Use markers: `@pytest.mark.integration` for slow tests

## Getting Help

- 📖 [Project README](README.md)
- 🐛 [Report Issues](https://github.com/KelvinDev945/tf_benchmark/issues)
- 💬 [Discussions](https://github.com/KelvinDev945/tf_benchmark/discussions)

## Summary

```bash
# One-time setup
./scripts/setup-dev-env.sh

# Before each commit
./scripts/validate-code.sh

# Commit (hooks run automatically)
git commit -m "Your message"
```

That's it! The scripts ensure your code meets all requirements. 🚀
