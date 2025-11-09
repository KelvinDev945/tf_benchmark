"""
Pytest configuration and fixtures.

This file is automatically loaded by pytest and provides:
- Path setup for importing src modules
- Common fixtures for tests
- Test environment configuration
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Check for optional dependencies
def _check_datasets_available():
    """Check if datasets library is available."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


HAS_DATASETS = _check_datasets_available()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_datasets: mark test as requiring HuggingFace datasets library",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests requiring optional dependencies."""
    skip_datasets = pytest.mark.skip(reason="requires datasets library (pip install datasets)")

    for item in items:
        # Skip tests that require datasets if not available
        if "requires_datasets" in item.keywords and not HAS_DATASETS:
            item.add_marker(skip_datasets)

        # Auto-detect ImageDatasetLoader tests
        if "ImageDatasetLoader" in item.nodeid and not HAS_DATASETS:
            item.add_marker(skip_datasets)
