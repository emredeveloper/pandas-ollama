# pandas-ollama

Natural language interface for pandas DataFrame analysis using Ollama models.

## Installation

```bash
pip install pandas-ollama
```

## Requirements

- Python 3.7+
- Ollama server running locally or remotely
- Required packages: pandas, matplotlib, seaborn, requests

## Quick Start

```python
import pandas as pd
from pandas_ollama import MyPandasAI

# Create a DataFrame
df = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet'],
    'Price': [1000, 800, 500],
    'Stock': [50, 100, 75]
})

# Create PandasOllama instance
panoll = MyPandasAI(df, model="llama3:latest")

# Ask a question about your data
result = panoll.ask("What is the average price of products?")
print(result.content)

# Create a visualization
result = panoll.plot("Show the distribution of prices", viz_type="bar")

# Display the visualization in a notebook
if result.visualization:
    import base64
    from IPython.display import Image
    image_data = base64.b64decode(result.visualization)
    Image(data=image_data)
```

## Features

- Natural language querying of pandas DataFrames
- Automatic visualization generation
- Support for various chart types: bar, line, scatter, hist, pie, heatmap, and more
- Intelligent data type detection
- Data transformation through natural language commands

## License

MIT License
