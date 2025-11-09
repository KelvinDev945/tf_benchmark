# GitHub Actions Workflows

This directory contains automated CI/CD workflows for the TensorFlow Benchmark project.

## 📋 Available Workflows

### 1. CI Workflow (`ci.yml`)

**Purpose**: Automated testing and coverage reporting

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`

**What it does**:
- ✅ Runs unit tests on Python 3.11 and 3.12
- ✅ Generates code coverage reports (XML, HTML, terminal)
- ✅ Uploads coverage to Codecov (requires `CODECOV_TOKEN` secret)
- ✅ Creates coverage artifacts for download
- ✅ Excludes integration and slow tests (runs only fast unit tests)

**Runtime**: ~3-5 minutes

**Configuration**:
- Test command: `pytest tests/ -v --cov=src`
- Test markers: Excludes `integration` and `slow` tests
- Coverage config: See `pyproject.toml`

---

### 2. Code Quality Workflow (`lint.yml`)

**Purpose**: Code quality and style checks

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`

**What it does**:
- ✅ **Black**: Code formatting check
- ✅ **isort**: Import sorting check
- ✅ **Flake8**: PEP-8 compliance and linting
- ⚠️ **mypy**: Type checking (non-blocking)

**Runtime**: ~2-3 minutes

**Configuration**:
- Black config: `pyproject.toml` (line length: 100)
- isort config: `pyproject.toml` (black-compatible profile)
- Flake8 config: `.flake8`
- mypy config: `pyproject.toml`

---

## 🚀 Running Checks Locally

Before pushing code, run these commands locally to catch issues early:

```bash
# Install code quality tools
pip install black flake8 isort mypy

# Format code
black src/ tests/
isort src/ tests/

# Run linters
flake8 src/ tests/
mypy src/

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## 🔧 Setup Requirements

### Required GitHub Secrets

For full functionality, configure these secrets in your repository settings:

| Secret Name | Purpose | Required |
|-------------|---------|----------|
| `CODECOV_TOKEN` | Upload coverage to Codecov | Optional |

**How to add secrets**:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add the secret name and value

---

## 📊 Viewing Results

### CI Workflow Results

- **Test Results**: Check the workflow run logs
- **Coverage Reports**:
  - View summary in workflow "Summary" tab
  - Download HTML coverage report from "Artifacts"
  - View on Codecov dashboard (if configured)

### Lint Workflow Results

- **Formatting Issues**: Listed in workflow logs with diffs
- **Linting Errors**: Displayed with file locations and error codes
- **Quick Fix Tips**: Summary includes commands to fix issues

---

## 🎯 Status Badges

Add these badges to your README.md to show workflow status:

```markdown
![CI](https://github.com/YOUR_USERNAME/tf_benchmark/workflows/CI/badge.svg)
![Code Quality](https://github.com/YOUR_USERNAME/tf_benchmark/workflows/Code%20Quality/badge.svg)
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 🔄 Workflow Configuration

### Matrix Strategy

The CI workflow uses a matrix strategy to test multiple Python versions:

```yaml
matrix:
  python-version: ["3.11", "3.12"]
```

To add more versions, edit the matrix in `ci.yml`.

### Caching

Both workflows use pip caching to speed up dependency installation:

```yaml
with:
  python-version: "3.11"
  cache: 'pip'
```

### Fail-Fast

CI workflow has `fail-fast: false` to run all matrix jobs even if one fails.

---

## 📝 Customization

### Excluding Specific Checks

**In Flake8** (`.flake8`):
```ini
per-file-ignores =
    __init__.py:F401
    tests/*:F401,F811
```

**In Black** (`pyproject.toml`):
```toml
extend-exclude = '''
/(
  # Add directories to exclude
  | custom_dir
)/
'''
```

### Adjusting Code Style

Edit `pyproject.toml` to change code style settings:

```toml
[tool.black]
line-length = 100  # Change to your preference

[tool.isort]
line_length = 100  # Should match black
```

---

## 🐛 Troubleshooting

### Workflow Fails on First Run

**Issue**: Dependencies not found or import errors

**Solution**: Ensure `requirements.txt` includes all test dependencies:
```bash
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.12.0
flake8>=6.1.0
isort>=5.13.0
mypy>=1.7.0
```

### Coverage Upload Fails

**Issue**: Codecov token not configured

**Solution**: Either add `CODECOV_TOKEN` secret or set `fail_ci_if_error: false` (already configured).

### Mypy Errors Block CI

**Issue**: Type checking is too strict for current codebase

**Solution**: Mypy is set to `continue-on-error: true` by default. Failures won't block CI.

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [mypy Documentation](https://mypy.readthedocs.io/)

---

**Last Updated**: 2025-11-08
