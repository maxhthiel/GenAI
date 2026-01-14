"""
This script defines a custom tool for Exploratory Data Analysis (EDA) within the smolagents framework.
It is designed to inspect financial datasets by extracting schema information and sample records to facilitate downstream processing.
"""

import pandas as pd
from smolagents import Tool

class EDASummaryTool(Tool):
    """
    A custom tool class that loads a CSV dataset and provides a summary of its structure.
    This tool allows the agent to inspect the dataset's schema (columns, types) and content before performing complex analysis.
    """
    name = "eda_summary"
    description = "Returns the column names, data types, and first 3 rows of the financial dataset. ALWAYS use this first to understand the data structure before writing any analysis code."
    inputs = {} # Kein Input nötig
    output_type = "string"

    def __init__(self, csv_path: str):
        """
        Initializes the tool instance and validates the data source.

        Args:
            csv_path (str): The file path to the CSV dataset.
        """
        super().__init__()
        self.csv_path = csv_path
        # Attempt to load the dataset immediately to validate file existence and cache the dataframe
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

    def forward(self):
        """
        Generates a summary of the dataset's structure and content.

        Returns:
            str: A formatted string containing the file path, column names, dimensions, and sample records.
        """
        # Construct a metadata dictionary to provide the agent with structural context for code generation
        info = {
            "columns": list(self.df.columns),
            # Convert pandas dtypes to strings to ensure compatibility with standard output formats
            "dtypes": {k: str(v) for k, v in self.df.dtypes.items()},
            "shape": self.df.shape,
            # Serialize the first 3 rows into a record-oriented dictionary for readability
            "sample_data": self.df.head(3).to_dict(orient="records"),
            # Important! path for the agent to load data ladter
            "file_path_for_pandas": self.csv_path 
        }
        
        return (
            f"DATASET INFO:\n"
            f"Path: {self.csv_path}\n" 
            f"Columns: {info['columns']}\n"
            f"Shape: {info['shape']}\n"
            f"Sample: {info['sample_data']}"
        )