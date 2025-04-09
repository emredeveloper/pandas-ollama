"""
pandas_ollama: Pandas verilerini doğal dil sorguları ile analiz eden AI kütüphanesi.
"""

from .core import MyPandasAI
from .response import StructuredResponse

__version__ = '1.0.1'
__all__ = ['MyPandasAI', 'StructuredResponse']