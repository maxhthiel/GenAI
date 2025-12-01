from smolagents import Tool
import pandas as pd

class EDASummaryTool(Tool):
    name = "eda_summary"
    description = "Provides a basic EDA summary of the CSV dataset (shape, columns, missing values)."
    inputs = {} # Kein Input nötig, da CSV fest geladen ist
    output_type = "string"

    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)

    def forward(self):
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isna().sum().to_dict(),
            "head": self.df.head(3).to_dict()
        }
        return str(summary)