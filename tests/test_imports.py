import sys
import os
import pytest
from pathlib import Path

# Add the project root to sys.path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """
    Simple test to verify that core modules can be imported without errors.
    This catches syntax errors and missing dependencies.
    """
    try:
        import tools
        import worker
        # main imports PySide6, which might not be available in headless CI environments 
        # without xvfb, but we can try importing it if dependencies are installed.
        # For now, let's stick to logic modules.
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_tools_functions():
    """
    Verify existence of key functions in tools.py
    """
    import tools
    assert hasattr(tools, 'extract_game_result')
    assert hasattr(tools, 'get_hammer')
