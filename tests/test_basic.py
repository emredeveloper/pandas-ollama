"""
Basic tests for pandas-ollama package.
"""

def test_import():
    """Test importing the main module."""
    import pandas_ollama
    from pandas_ollama import MyPandasAI
    assert hasattr(pandas_ollama, '__version__')

def test_version():
    """Test version exists and is a string."""
    from pandas_ollama import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    assert __version__ == '1.0.4'  # Updated version number
