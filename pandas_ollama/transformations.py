"""
Module for data transformation operations - Compatible with CSV and general data
"""

import pandas as pd
import re
import json
from typing import Dict, Tuple, List, Any, Optional

class DataTransformer:
    """Pandas DataFrame transformation class - Compatible with any data structure"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        # Supported data transformations
        self.supported_transforms = {
            "filter": self.transform_filter,
            "sort": self.transform_sort,
            "group": self.transform_group,
            "aggregate": self.transform_aggregate,
            "pivot": self.transform_pivot,
            "select": self.transform_select,  # New: Select specific columns
            "rename": self.transform_rename,  # New: Rename columns
            "fillna": self.transform_fillna,  # New: Fill missing values
            "dropna": self.transform_dropna   # New: Drop missing values
        }
        
        # Information about data types
        self._analyze_datatypes()
    
    def _analyze_datatypes(self) -> None:
        """Collects information about data types in the DataFrame"""
        self.numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = self.df.select_dtypes(include=['datetime']).columns.tolist()
        self.bool_columns = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        # Detect potential key columns (with unique values)
        self.key_columns = []
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df) and self.df[col].nunique() > 1:
                self.key_columns.append(col)
    
    def parse_transform_response(self, response: str) -> Tuple[str, Dict, str]:
        """Parses transformation response"""
        transform_type = ""
        params = {}
        code = ""
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if "TYPE:" in line:
                current_section = "type"
                transform_type = line.replace("TYPE:", "").strip().lower()
            elif "PARAMETERS:" in line:
                current_section = "params"
                param_text = line.replace("PARAMETERS:", "").strip()
                try:
                    # Try to parse parameters in JSON format
                    if param_text.startswith("{") and param_text.endswith("}"):
                        import json
                        params = json.loads(param_text)
                except:
                    pass
            elif "CODE:" in line:
                current_section = "code"
                continue
            
            if current_section == "code":
                code += line + "\n"
        
        # Smart detection for general filtering cases
        if not transform_type or not params:
            transform_type, params = self._detect_transformation_from_text(response)
        
        return transform_type, params, code.strip()
    
    def _detect_transformation_from_text(self, text: str) -> Tuple[str, Dict]:
        """Detects transformation type and parameters from text"""
        text = text.lower()
        
        # Filter operation detection
        if any(x in text for x in ["filter", "where", "select", "find", "get", "show"]):
            column, condition, value = self._extract_filter_params(text)
            if column:
                return "filter", {"column": column, "condition": condition, "value": value}
        
        # Sort operation detection
        if any(x in text for x in ["sort", "order", "arrange"]):
            column, ascending = self._extract_sort_params(text)
            if column:
                return "sort", {"column": column, "ascending": ascending}
        
        # Group operation detection
        if any(x in text for x in ["group", "aggregate"]):
            columns, agg_func = self._extract_group_params(text)
            if columns:
                return "group", {"columns": columns, "agg_func": agg_func}
        
        # Column selection operation detection
        if any(x in text for x in ["select column", "columns"]):
            columns = self._extract_columns_from_text(text)
            if columns:
                return "select", {"columns": columns}
        
        # Missing value operation detection
        if any(x in text for x in ["missing", "null", "na", "empty"]):
            if any(x in text for x in ["fill", "replace"]):
                return "fillna", {"value": 0}  # Default fill with 0
            elif any(x in text for x in ["drop", "remove", "delete"]):
                return "dropna", {"how": "any"}
                
        return "", {}
    
    def _extract_filter_params(self, text: str) -> Tuple[str, str, Any]:
        """Extracts filtering parameters from text"""
        # Check for all columns
        column = None
        best_match_pos = float('inf')
        
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in text:
                # If multiple columns match, choose the one that appears earlier
                col_pos = text.find(col_lower)
                if col_pos < best_match_pos:
                    column = col
                    best_match_pos = col_pos
        
        # If no column found, try to guess the most appropriate column
        if not column:
            # Compare words in text with column names
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if len(word) > 3:  # Check words with at least 4 characters
                    for col in self.df.columns:
                        if word.lower() in col.lower():
                            column = col
                            break
                    if column:
                        break
        
        # If still not found and looks like a numeric filter, use first numeric column
        if not column and re.search(r'\d+', text) and self.numeric_columns:
            column = self.numeric_columns[0]
        
        # Find condition operator
        condition = "=="  # Default equality
        
        # Check comparison operators
        if any(x in text for x in ["greater", "more", "above", "over", ">"]):
            condition = ">"
        elif any(x in text for x in ["less", "fewer", "below", "under", "<"]):
            condition = "<"
        elif any(x in text for x in ["not equal", "outside", "except", "!="]):
            condition = "!="
        elif any(x in text for x in ["greater equal", "at least", "minimum", ">="]):
            condition = ">="
        elif any(x in text for x in ["less equal", "at most", "maximum", "<="]):
            condition = "<="
        elif any(x in text for x in ["contains", "including", "has", "with"]):
            condition = "contains"
        
        # Find value
        value = None
        
        # Find numerical value
        if column and column in self.numeric_columns:
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if numbers:
                # Choose the most appropriate number (close to condition operator)
                cond_terms = ["greater", "less", "equal", "more", "fewer"]
                best_num_pos = float('inf')
                best_num = None
                
                for num in numbers:
                    for term in cond_terms:
                        term_pos = text.find(term)
                        if term_pos > -1:
                            num_pos = text.find(num)
                            distance = abs(num_pos - term_pos)
                            if distance < best_num_pos:
                                best_num_pos = distance
                                best_num = num
                
                if best_num:
                    if '.' in best_num:
                        value = float(best_num)
                    else:
                        value = int(best_num)
                else:
                    value = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
        
        # Find categorical value
        elif column and column in self.categorical_columns:
            # Get possible values
            possible_values = self.df[column].unique()
            
            # Find value in text
            for val in possible_values:
                if str(val).lower() in text:
                    value = val
                    break
            
            # If not found, try a common word in the text
            if value is None:
                words = re.findall(r'\b\w+\b', text)
                word_counts = {}
                for word in words:
                    if len(word) > 3:  # Words longer than 3 characters
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                # Find most common word
                if word_counts:
                    most_common = max(word_counts.items(), key=lambda x: x[1])[0]
                    value = most_common
        
        # If value still not found, use a default
        if value is None and column:
            if column in self.numeric_columns:
                # Average value for numerical column
                value = self.df[column].mean()
            elif column in self.categorical_columns:
                # Most frequent value for categorical column
                value = self.df[column].mode()[0]
        
        return column, condition, value
    
    def _extract_sort_params(self, text: str) -> Tuple[str, bool]:
        """Extracts sorting parameters from text"""
        column = None
        
        # Find column name - try exact matches first
        for col in self.df.columns:
            if col.lower() in text.lower():
                column = col
                break
        
        # If not found, try to find a numerical column
        if not column and self.numeric_columns:
            # If text contains "descending" or "decreasing", probably a numerical column is wanted
            if any(x in text.lower() for x in ["descending", "decreasing", "desc"]):
                column = self.numeric_columns[0]
            else:
                column = self.numeric_columns[0]
        
        # If still not found, use any column
        if not column and len(self.df.columns) > 0:
            column = self.df.columns[0]
        
        # Find sort direction
        ascending = True  # Default ascending
        if any(x in text.lower() for x in ["descending", "decreasing", "desc", "reverse"]):
            ascending = False
        
        return column, ascending
    
    def _extract_group_params(self, text: str) -> Tuple[list, str]:
        """Extracts grouping parameters from text"""
        columns = []
        
        # Check categorical columns first
        for col in self.categorical_columns:
            if col.lower() in text.lower():
                columns.append(col)
        
        # If no categorical column found, try any column
        if not columns:
            for col in self.df.columns:
                if col.lower() in text.lower():
                    columns.append(col)
                    break
        
        # If still not found and categorical columns exist, use the first one
        if not columns and self.categorical_columns:
            columns = [self.categorical_columns[0]]
        
        # Find aggregation function
        agg_func = "mean"  # Default average
        
        if any(x in text.lower() for x in ["sum", "total"]):
            agg_func = "sum"
        elif any(x in text.lower() for x in ["count", "number", "quantity"]):
            agg_func = "count"
        elif any(x in text.lower() for x in ["max", "largest", "maximum", "highest"]):
            agg_func = "max"
        elif any(x in text.lower() for x in ["min", "smallest", "minimum", "lowest"]):
            agg_func = "min"
        elif any(x in text.lower() for x in ["mean", "average", "avg"]):
            agg_func = "mean"
        
        return columns, agg_func
    
    def _extract_columns_from_text(self, text: str) -> List[str]:
        """Extracts column names from text"""
        columns = []
        
        # Check all columns
        for col in self.df.columns:
            if col.lower() in text.lower():
                columns.append(col)
        
        # If no columns found, check for words in the text
        if not columns:
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                if len(word) > 3:  # Check words with at least 4 characters
                    for col in self.df.columns:
                        if word.lower() in col.lower() or col.lower() in word.lower():
                            columns.append(col)
        
        # Remove duplicates
        columns = list(dict.fromkeys(columns))
        
        return columns
    
    def transform_filter(self, params: Dict) -> pd.DataFrame:
        """Filter the DataFrame"""
        column = params.get("column")
        condition = params.get("condition", "==")
        value = params.get("value")
        
        if not column or value is None or column not in self.df.columns:
            return self.df
            
        if condition == "==":
            return self.df[self.df[column] == value]
        elif condition == "!=":
            return self.df[self.df[column] != value]
        elif condition == ">":
            return self.df[self.df[column] > value]
        elif condition == "<":
            return self.df[self.df[column] < value]
        elif condition == ">=":
            return self.df[self.df[column] >= value]
        elif condition == "<=":
            return self.df[self.df[column] <= value]
        elif condition == "in":
            if isinstance(value, list):
                return self.df[self.df[column].isin(value)]
        elif condition == "contains":
            # Works for both strings and lists
            if isinstance(value, str):
                # If column is string type, use contains
                if self.df[column].dtype == 'object':
                    return self.df[self.df[column].str.contains(str(value), na=False, case=False)]
                else:
                    # Otherwise convert to string and check
                    return self.df[self.df[column].astype(str).str.contains(str(value), na=False, case=False)]
        
        return self.df

    def transform_sort(self, params: Dict) -> pd.DataFrame:
        """Sort the DataFrame"""
        column = params.get("column")
        ascending = params.get("ascending", True)
        
        if not column or column not in self.df.columns:
            return self.df
            
        return self.df.sort_values(by=column, ascending=ascending)

    def transform_group(self, params: Dict) -> pd.DataFrame:
        """Group the DataFrame"""
        columns = params.get("columns")
        agg_func = params.get("agg_func", "mean")
        
        if not columns:
            return self.df
            
        if isinstance(columns, str):
            columns = [columns]
        
        # Check column existence
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            return self.df
            
        # Automatically detect which columns to aggregate
        if agg_func in ["mean", "sum", "min", "max"]:
            # Use numerical columns
            agg_columns = [col for col in self.numeric_columns if col not in valid_columns]
            if not agg_columns:
                # If no numerical columns, use count
                return self.df.groupby(valid_columns).size().reset_index(name='count')
            
            # Create aggregation dictionary
            agg_dict = {col: agg_func for col in agg_columns}
            return self.df.groupby(valid_columns).agg(agg_dict).reset_index()
        else:
            # For other aggregation functions like count
            return self.df.groupby(valid_columns).agg(agg_func).reset_index()

    # New transformation functions
    def transform_select(self, params: Dict) -> pd.DataFrame:
        """Select specific columns"""
        columns = params.get("columns")
        
        if not columns:
            return self.df
            
        if isinstance(columns, str):
            columns = [columns]
        
        # Check column existence
        valid_columns = [col for col in columns if col in self.df.columns]
        if not valid_columns:
            return self.df
            
        return self.df[valid_columns]
    
    def transform_rename(self, params: Dict) -> pd.DataFrame:
        """Rename columns"""
        rename_dict = params.get("rename_dict", {})
        
        if not rename_dict:
            return self.df
            
        # Check column existence
        valid_renames = {old: new for old, new in rename_dict.items() if old in self.df.columns}
        if not valid_renames:
            return self.df
            
        return self.df.rename(columns=valid_renames)
    
    def transform_fillna(self, params: Dict) -> pd.DataFrame:
        """Fill missing values"""
        value = params.get("value")
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Check column existence
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            # Fill only specified columns
            result = self.df.copy()
            for col in valid_columns:
                result[col] = result[col].fillna(value)
            return result
        else:
            # Fill entire DataFrame
            return self.df.fillna(value)
    
    def transform_dropna(self, params: Dict) -> pd.DataFrame:
        """Drop missing values"""
        how = params.get("how", "any")  # 'any' or 'all'
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Check column existence
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            # Drop rows with missing values in specified columns
            return self.df.dropna(subset=valid_columns, how=how)
        else:
            # Drop rows with missing values in all columns
            return self.df.dropna(how=how)
    
    # Existing transformation functions...
    def transform_aggregate(self, params: Dict) -> pd.DataFrame:
        """Perform aggregation operations on DataFrame"""
        agg_func = params.get("agg_func", "mean")
        columns = params.get("columns")
        
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            # Check column existence
            valid_columns = [col for col in columns if col in self.df.columns]
            if not valid_columns:
                return self.df
                
            return self.df[valid_columns].agg(agg_func).to_frame().reset_index()
        else:
            # Automatically use numerical columns
            return self.df[self.numeric_columns].agg(agg_func).to_frame().reset_index()

    def transform_pivot(self, params: Dict) -> pd.DataFrame:
        """Create a pivot table"""
        index = params.get("index")
        columns = params.get("columns")
        values = params.get("values")
        
        if not index or not columns or not values:
            return self.df
            
        # Check column existence
        if index not in self.df.columns or columns not in self.df.columns or values not in self.df.columns:
            return self.df
            
        try:
            return self.df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean').reset_index()
        except:
            # Return original data if pivot fails
            return self.df
