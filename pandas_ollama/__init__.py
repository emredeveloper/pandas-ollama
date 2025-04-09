"""
pandas_ollama: Pandas verilerini doğal dil sorguları ile analiz eden AI kütüphanesi.

Bu kütüphane, pandas DataFrame'leri üzerinde doğal dil sorguları yaparak
veri analizi, dönüşümü ve görselleştirme işlemlerini kolaylaştırır.

Temel Kullanım:
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
    # Google Colab bağımlılıkları yerel ortamda olmayabilir
    pass

__version__ = '1.0.3'
__author__ = 'Cihat Emre Karataş'
__all__ = ['MyPandasAI', 'StructuredResponse', 'OllamaColabAdapter']