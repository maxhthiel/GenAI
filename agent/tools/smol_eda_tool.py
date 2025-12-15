import pandas as pd
from smolagents import Tool

class EDASummaryTool(Tool):
    name = "eda_summary"
    description = "Returns the column names, data types, and first 3 rows of the financial dataset. ALWAYS use this first to understand the data structure before writing any analysis code."
    inputs = {} # Kein Input nötig
    output_type = "string"

    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        # Testen ob Datei existiert
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

    def forward(self):
        # Wir geben Metadaten zurück, damit der Agent weiß, wie er coden soll
        info = {
            "columns": list(self.df.columns),
            "dtypes": {k: str(v) for k, v in self.df.dtypes.items()},
            "shape": self.df.shape,
            "sample_data": self.df.head(3).to_dict(orient="records"),
            "file_path_for_pandas": self.csv_path # <--- WICHTIG!
        }
        
        return (
            f"DATASET INFO:\n"
            f"Path: {self.csv_path}\n" # Agent sieht den Pfad und kann ihn nutzen
            f"Columns: {info['columns']}\n"
            f"Shape: {info['shape']}\n"
            f"Sample: {info['sample_data']}"
        )