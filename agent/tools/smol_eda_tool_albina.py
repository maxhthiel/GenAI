import pandas as pd
from smolagents import Tool

class EDAQueryTool(Tool):
    name = "eda_query"
    description = """
Use this tool for ANY question related to the CSV dataset.

You MUST call this tool whenever:
- the user asks about filters (e.g., PE Ratio between X and Y)
- describes statistical analysis
- requests numbers from the dataset
- compares companies
- asks about correlations or distributions
- requests summarization of a column

Rules:
1. ALWAYS execute Python code using the variable `df` (pandas DataFrame).
2. ALWAYS assign the final output to a variable named `result`.
3. NEVER answer dataset-related questions yourself.
4. NEVER use external knowledge. Only use `df`.
"""

    inputs = {
        "code": {
            "type": "string",
            "description": "Python code to execute using `df`. MUST assign the final output to variable `result`."
        }
    }

    output_type = "string"

    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        print(f"[EDAQueryTool] Loaded CSV: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def forward(self, code: str):
        print("\n[EDAQueryTool] Executing code:")
        print(code)
        print("-" * 50)

        local_env = {"df": self.df, "pd": pd}

        try:
            exec(code, {}, local_env)
            result = local_env.get("result", None)

            if result is None:
                return "ERROR: No variable `result` returned by the code."

            return str(result)

        except Exception as e:
            return f"Error executing EDA code: {e}"

