import taskpriors

def test_version_string():
    """__version__ should be a non-empty string matching semantic versioning."""
    version = taskpriors.__version__
    assert isinstance(version, str)
    parts = version.split('.')
    # Expect at least major.minor.patch
    assert len(parts) == 3 and all(p.isdigit() for p in parts)

def test_analyze_exported():
    """analyze should be exported in the package __all__ and be callable."""
    assert 'analyze' in taskpriors.__all__
    assert callable(taskpriors.analyze)