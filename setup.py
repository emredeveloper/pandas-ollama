from setuptools import setup, find_packages
import os

# Read the contents of your README file
readme_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Pandas-Ollama: Natural language interface for pandas DataFrame analysis using Ollama models"

# Get requirements from requirements.txt
requirements_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'pandas_ollama', 'requirements.txt'
)
try:
    with open(requirements_path, 'r') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    # Fallback requirements
    requirements = [
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "typing-extensions>=4.0.0",
    ]

setup(
    name="pandas-ollama",
    version="1.0.4",  # Updated version
    author="Cihat Emre Karatas",
    author_email="chtemrekrts@gmail.com",
    description="Natural language interface for pandas DataFrame analysis using Ollama models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emredeveloper/pandas-ollama",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",  # Python 3.8+ is recommended as 3.7 is nearing EOL
    install_requires=requirements,
    include_package_data=True,
    keywords="pandas, ollama, ai, dataframe, analysis, visualization, natural language",
    project_urls={
        "Bug Reports": "https://github.com/emredeveloper/pandas-ollama/issues",
        "Source": "https://github.com/emredeveloper/pandas-ollama",
    },
)
