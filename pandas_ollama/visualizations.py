"""
Module containing functions for data visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import traceback
import signal
import warnings
import re
from typing import Optional, Dict, List, Callable, Any
from contextlib import contextmanager
from datetime import datetime

class TimeoutException(Exception):
    """Custom exception class for timeout errors"""
    pass

@contextmanager
def time_limit(seconds):
    """Terminates the process after a specified time"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timeout reached!")
        
    # Signal only works on Unix-based systems (Linux, macOS)
    # Doesn't work on Windows, so we call it inside a try-except block
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        yield
    except (ValueError, AttributeError):
        # Silently pass on Windows or systems without signal support
        yield
    finally:
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass

class Visualizer:
    """Pandas DataFrame visualization class"""
    
    def __init__(self, dataframe: pd.DataFrame, timeout: int = 30):
        self.df = dataframe
        self.timeout = timeout
        
        # Determine column types
        self._analyze_datatypes()
        
        # Supported visualizations
        self.supported_viz = {
            "bar": self.plot_bar,
            "line": self.plot_line,
            "scatter": self.plot_scatter,
            "hist": self.plot_histogram,
            "pie": self.plot_pie,
            "heatmap": self.plot_heatmap,
            "box": self.plot_boxplot,
            "count": self.plot_countplot,   # New: Categorical count plot
            "area": self.plot_area,         # New: Area plot
            "density": self.plot_density,   # New: Density plot
            "violin": self.plot_violin      # New: Violin plot
        }
    
    def _detect_date_format(self, sample_dates):
        """
        Attempts to detect date format from data samples
        """
        # Common date formats
        common_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
            '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y'
        ]
        
        # Formats including time
        time_formats = [
            '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        
        # Combine default formats
        all_formats = common_formats + time_formats
        
        for date_str in sample_dates:
            if not isinstance(date_str, str) or pd.isna(date_str):
                continue
                
            # Test each format
            for fmt in all_formats:
                try:
                    datetime.strptime(date_str, fmt)
                    return fmt
                except:
                    pass
        
        # If unable to detect date format, look for patterns similar to strftime format
        date_pattern = r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$"  # YYYY-MM-DD, DD/MM/YYYY etc.
        datetime_pattern = r"^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\s\d{1,2}:\d{1,2}(:\d{1,2})?$"  # YYYY-MM-DD HH:MM:SS
        
        # Test patterns for sample dates
        for date_str in sample_dates:
            if not isinstance(date_str, str) or pd.isna(date_str):
                continue
                
            if re.match(datetime_pattern, date_str):
                # Format with time
                if '-' in date_str:
                    return '%Y-%m-%d %H:%M:%S' if date_str[0:4].isdigit() else '%d-%m-%Y %H:%M:%S'
                elif '/' in date_str:
                    return '%Y/%m/%d %H:%M:%S' if date_str[0:4].isdigit() else '%d/%m/%Y %H:%M:%S'
                elif '.' in date_str:
                    return '%Y.%m.%d %H:%M:%S' if date_str[0:4].isdigit() else '%d.%m.%Y %H:%M:%S'
            elif re.match(date_pattern, date_str):
                # Date-only format
                if '-' in date_str:
                    return '%Y-%m-%d' if date_str[0:4].isdigit() else '%d-%m-%Y'
                elif '/' in date_str:
                    return '%Y/%m/%d' if date_str[0:4].isdigit() else '%d/%m/%Y'
                elif '.' in date_str:
                    return '%Y.%m.%d' if date_str[0:4].isdigit() else '%d.%m.%Y'
        
        return None
    
    def _is_likely_date_column(self, column_name):
        """
        Estimates the likelihood of a column being a date column from its name
        """
        name_lower = column_name.lower()
        date_keywords = ['date', 'time', 'year', 'month', 'day']
        
        for keyword in date_keywords:
            if keyword in name_lower:
                return True
        return False
    
    def _analyze_datatypes(self) -> None:
        """Collects information about data types in the DataFrame"""
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Detect date columns
        self.datetime_columns = []
        
        # First add columns already of datetime type
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_columns.append(col)
        
        # Then check columns that are strings but might be in date format
        for col in self.categorical_columns:
            if col not in self.datetime_columns and (self._is_likely_date_column(col) or 
               any(isinstance(s, str) and ('/' in s or '-' in s) for s in self.df[col].head().values)):
                try:
                    # Try to detect date format
                    sample_values = self.df[col].dropna().head(10).values
                    date_format = self._detect_date_format(sample_values)
                    
                    # If format detected, use that format
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if date_format:
                            pd.to_datetime(self.df[col], format=date_format, errors='raise')
                        else:
                            pd.to_datetime(self.df[col], errors='raise')
                            
                    self.datetime_columns.append(col)
                except:
                    pass
        
        # Detect boolean columns
        self.bool_columns = self.df.select_dtypes(include=['bool']).columns.tolist()
    
    def _get_best_columns_for_visualization(self, viz_type: str, query: str = "") -> Dict:
        """Selects the most suitable columns for visualization"""
        params = {}
        
        if viz_type == "scatter":
            # Scatter plot requires two numerical columns
            if len(self.numeric_columns) >= 2:
                params["x"] = self.numeric_columns[0]
                params["y"] = self.numeric_columns[1]
        elif viz_type == "hist":
            # Histogram requires a numerical column
            if self.numeric_columns:
                params["x"] = self.numeric_columns[0]
        elif viz_type == "pie":
            # Pie chart requires a categorical column
            if self.categorical_columns:
                params["names"] = self.categorical_columns[0]
        elif viz_type == "heatmap":
            # Heatmap for correlation
            if len(self.numeric_columns) > 1:
                params["data"] = "correlation"
                params["columns"] = self.numeric_columns
        elif viz_type == "box":
            # Box plot requires categorical and numerical columns
            if self.categorical_columns and self.numeric_columns:
                params["x"] = self.categorical_columns[0]
                params["y"] = self.numeric_columns[0]
        
        return params
    
    def _clean_code(self, code: str) -> str:
        """
        Cleans markdown formatted code blocks
        - Removes ```python and ``` blocks
        - Removes single backticks
        """
        if not code:
            return ""
            
        # Remove ```python and ``` patterns
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        
        # Remove single backticks at the beginning of line or at start
        code = re.sub(r'(^|[\n])(`)([\w\s])', r'\1\3', code)
        
        # Remove unnecessary headings if present
        code = re.sub(r'^(Run this code|You can use this code|Python code:|Code:|Visual:|Visualization code:).*[\n]', '', code, flags=re.MULTILINE)
        
        return code
    
    def execute_code(self, code: str, timeout: int = None) -> Optional[str]:
        """
        Safely executes the given visualization code and returns a base64 encoded image.
        
        Args:
            code (str): Python code to execute
            timeout (int, optional): Process timeout in seconds
            
        Returns:
            Optional[str]: Base64 encoded image data
        """
        if not code:
            return None
        
        # Clean markdown format
        code = self._clean_code(code)
        
        # If timeout not specified, use instance value
        if timeout is None:
            timeout = self.timeout
            
        try:
            # Clean code and perform security check
            if "import " in code and not any(x in code for x in ["import matplotlib", "import seaborn", "import pandas", "import numpy"]):
                # For security, only allow specific imports
                raise ValueError("For security reasons, only matplotlib, pandas, numpy and seaborn libraries can be used.")
            
            # Adjust code to use existing DataFrame
            modified_code = code.replace("df = pd.DataFrame", "# df is already defined")
            
            # Create image buffer
            buffer = io.BytesIO()
            
            # Prepare locals and globals dictionaries
            import numpy as np
            loc = {
                "plt": plt, 
                "sns": sns, 
                "pd": pd, 
                "np": np,
                "df": self.df, 
                "buffer": buffer
            }
            
            # Execute code with timeout
            with time_limit(timeout):
                # Execute code (should create matplotlib figure by default)
                print(f"⚙️ Executing visualization code (timeout: {timeout} seconds)...")
                
                # If code doesn't contain fig, figure or plt.figure() usage
                # create a default figure
                if not re.search(r'(plt\.figure|fig\s*=|figure\s*=)', modified_code):
                    plt.figure(figsize=(10, 6))
                    
                exec(modified_code, globals(), loc)
                
                # Ensure the figure is saved
                try:
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    plt.close('all')  # Close all figures
                except Exception as e:
                    print(f"⚠️ Figure save error: {str(e)}")
                    # Maybe code already saved or didn't create a figure
                    pass
                
                # Base64 encode
                buffer.seek(0)
                image_data = buffer.read()
                
                # If buffer is empty or invalid, raise error
                if not image_data or len(image_data) < 100:  # Very small files are likely invalid
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, "Visualization could not be created", 
                             horizontalalignment='center', verticalalignment='center', fontsize=14)
                    plt.axis('off')
                    plt.savefig(buffer, format='png')
                    plt.close()
                    buffer.seek(0)
                    image_data = buffer.read()
                
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                return image_base64
                
        except TimeoutException:
            print(f"⚠️ Visualization timed out (> {timeout} seconds)")
            # Create error image for timeout
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Visualization timed out (> {timeout} seconds)", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            traceback.print_exc()
            
            # Create error image
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

    def plot_bar(self, query: str) -> Optional[str]:
        """Creates a bar chart"""
        try:
            if not self.numeric_columns or not self.categorical_columns:
                return self._create_error_viz("No suitable columns found")
                
            x = self.categorical_columns[0]
            y = self.numeric_columns[0]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Bar Chart")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Bar chart error: {str(e)}")
            return self._create_error_viz(f"Bar chart error: {str(e)}")

    def _create_error_viz(self, error_message: str) -> str:
        """Creates a visualization with error message"""
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, error_message, 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    # New visualization types
    def plot_countplot(self, query: str) -> Optional[str]:
        """Creates a plot showing counts of categorical columns"""
        try:
            if not self.categorical_columns:
                return self._create_error_viz("No categorical column found")
                
            cat_col = self.categorical_columns[0]
            
            plt.figure(figsize=(10, 6))
            sns.countplot(x=cat_col, data=self.df)
            plt.title(f"{cat_col} Value Counts")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Count plot error: {str(e)}")
            return self._create_error_viz(f"Count plot error: {str(e)}")

    def plot_area(self, query: str) -> Optional[str]:
        """Creates an area chart"""
        try:
            if not self.numeric_columns or len(self.numeric_columns) < 2:
                return self._create_error_viz("Not enough numerical columns found")
                
            plt.figure(figsize=(12, 6))
            self.df[self.numeric_columns].plot.area(alpha=0.5)
            plt.title("Numerical Variables Area Chart")
            plt.grid(True)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Area chart error: {str(e)}")
            return self._create_error_viz(f"Area chart error: {str(e)}")

    def plot_density(self, query: str) -> Optional[str]:
        """Creates a density plot (KDE)"""
        try:
            if not self.numeric_columns:
                return self._create_error_viz("No numerical column found")
                
            plt.figure(figsize=(10, 6))
            for col in self.numeric_columns[:3]:  # Show max 3 columns
                sns.kdeplot(self.df[col], label=col)
            plt.title("Numerical Variables Density Plot")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Density plot error: {str(e)}")
            return self._create_error_viz(f"Density plot error: {str(e)}")

    def plot_violin(self, query: str) -> Optional[str]:
        """Creates a violin plot"""
        try:
            if not self.numeric_columns or not self.categorical_columns:
                return self._create_error_viz("No suitable columns found")
                
            x = self.categorical_columns[0]
            y = self.numeric_columns[0]
            
            plt.figure(figsize=(12, 6))
            sns.violinplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Violin Plot")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Violin plot error: {str(e)}")
            return self._create_error_viz(f"Violin plot error: {str(e)}")

    def plot_line(self, query: str) -> Optional[str]:
        """Creates a line chart"""
        try:
            # Check if suitable for temporal data
            if self.datetime_columns:
                x = self.datetime_columns[0]
                
                # Find first numerical column
                y = None
                for col in self.numeric_columns:
                    y = col
                    break
                
                if not y:
                    return self._create_error_viz("No numerical column found")
                
                plt.figure(figsize=(12, 6))
                plt.plot(self.df[x], self.df[y], marker='o', linestyle='-')
                plt.title(f"{y} Time Series")
                plt.grid(True)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            
            # If no temporal data, use index or sequential numerical values
            elif self.numeric_columns:
                # Determine most suitable columns
                y = self.numeric_columns[0]
                
                plt.figure(figsize=(12, 6))
                plt.plot(self.df.index, self.df[y], marker='o', linestyle='-')
                plt.title(f"{y} Series Plot")
                plt.grid(True)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
                
            return self._create_error_viz("No suitable column found for visualization")
        except Exception as e:
            print(f"Line chart error: {str(e)}")
            return self._create_error_viz(f"Line chart error: {str(e)}")

    def plot_scatter(self, query: str) -> Optional[str]:
        """Creates a scatter plot"""
        try:
            # Determine most suitable columns
            params = self._get_best_columns_for_visualization("scatter", query)
            
            if not params or "x" not in params or "y" not in params:
                return self._create_error_viz("No numerical columns found")
                
            x = params["x"]
            y = params["y"]
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Scatter Plot")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Scatter plot error: {str(e)}")
            return self._create_error_viz(f"Scatter plot error: {str(e)}")

    def plot_histogram(self, query: str) -> Optional[str]:
        """Creates a histogram"""
        try:
            # Determine most suitable columns
            params = self._get_best_columns_for_visualization("hist", query)
            
            if not params or "x" not in params:
                return self._create_error_viz("No numerical column found")
                
            x = params["x"]
            
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[x], kde=True)
            plt.title(f"{x} Distribution")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Histogram error: {str(e)}")
            return self._create_error_viz(f"Histogram error: {str(e)}")

    def plot_pie(self, query: str) -> Optional[str]:
        """Creates a pie chart"""
        try:
            # Determine most suitable columns
            params = self._get_best_columns_for_visualization("pie", query)
            
            if not params or "names" not in params:
                return self._create_error_viz("No categorical column found")
                
            names = params["names"]
            
            counts = self.df[names].value_counts()
            
            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title(f"{names} Distribution")
            plt.axis('equal')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Pie chart error: {str(e)}")
            return self._create_error_viz(f"Pie chart error: {str(e)}")

    def plot_heatmap(self, query: str) -> Optional[str]:
        """Creates a heatmap"""
        try:
            # Determine most suitable columns
            params = self._get_best_columns_for_visualization("heatmap", query)
            
            if not params or "data" not in params or params["data"] != "correlation":
                return self._create_error_viz("Not enough numerical columns for correlation")
                
            columns = params["columns"]
            
            plt.figure(figsize=(12, 10))
            correlation = self.df[columns].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Heatmap error: {str(e)}")
            return self._create_error_viz(f"Heatmap error: {str(e)}")

    def plot_boxplot(self, query: str) -> Optional[str]:
        """Creates a box plot"""
        try:
            # Determine most suitable columns
            params = self._get_best_columns_for_visualization("box", query)
            
            if not params or "x" not in params or "y" not in params:
                return self._create_error_viz("No suitable columns found")
                
            x = params["x"]
            y = params["y"]
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=x, y=y, data=self.df)
            plt.title(f"{x} - {y} Box Plot")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Box plot error: {str(e)}")
            return self._create_error_viz(f"Box plot error: {str(e)}")
