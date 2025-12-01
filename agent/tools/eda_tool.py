import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


class EDATool:
    """
    Ein simples Tool für grundlegende EDA.
    Es lädt die CSV einmal im Konstruktor und
    stellt einfache Analysefunktionen bereit.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def basic_summary(self):
        """
        Gibt Head, Describe, Missing Values etc. zurück.
        Ideal für den Start der Datenanalyse.
        """
        return {
            "head": self.df.head().to_dict(),
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isna().sum().to_dict(),
            "describe": self.df.describe().to_dict()
        }

    def plot_distribution(self, column):
        """
        Erstellt einen Histogrammplot als Base64-encoded PNG.
        Damit kannst du ihn überall einbetten (Notebook/Frontend).
        """
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df[column], kde=True)
        plt.title(f"Distribution of {column}")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        encoded = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return encoded
