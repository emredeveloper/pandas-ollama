"""
pandas_ollama: AI library for analyzing pandas DataFrames using natural language queries.

This library simplifies data analysis, transformation, and visualization 
operations on pandas DataFrames through natural language queries.

Basic Usage:
    >>> import pandas as pd
    >>> from pandas_ollama import MyPandasAI
    >>> df = pd.DataFrame({'Product': ['A', 'B', 'C'], 'Price': [10, 20, 30]})
    >>> ai = MyPandasAI(df, model="llama3:latest")
    >>> result = ai.ask("What is the average price?")
    >>> print(result.content)

GitHub: https://github.com/emredeveloper/pandas-ollama
"""

from .core import MyPandasAI
from .response import StructuredResponse
try:
    from .colab_adapter import OllamaColabAdapter
except ImportError:
    # Google Colab dependencies might not be available in local environment
    pass

__version__ = '1.0.4'
__author__ = 'Cihat Emre Karatas'
__all__ = ['MyPandasAI', 'StructuredResponse', 'OllamaColabAdapter']