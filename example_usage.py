import pandas as pd
import matplotlib.pyplot as plt
from pandas_ollama import MyPandasAI

# Create a DataFrame
df = pd.DataFrame({
    'Product': ['Laptop', 'Phone', 'Tablet'],
    'Price': [1000, 800, 500],
    'Stock': [50, 100, 75]
})

# Create PandasOllama instance
panoll = MyPandasAI(df, model="qwen2.5:7b")

# Ask a question about your data
result = panoll.ask("What is the average price of products?")
print(result.content)

# Create a visualization
result = panoll.plot("Show the distribution of prices", viz_type="bar")
print(result.content)

# Save the visualization to a file
if result.visualization:
    import base64
    with open("price_distribution.png", "wb") as f:
        f.write(base64.b64decode(result.visualization))
    print("Visualization saved as price_distribution.png")

# For Jupyter notebooks, display the visualization:
"""
# To display in Jupyter:
if result.visualization:
    import base64
    from IPython.display import display, Image
    image_data = base64.b64decode(result.visualization)
    display(Image(data=image_data))
"""
