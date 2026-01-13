# 1. Basis-Image: Schlankes Python für schnellere Bauzeiten
FROM python:3.11-slim

# 2. System-Abhängigkeiten für ChromaDB, Pandas und Bildverarbeitung
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/

# 3. Arbeitsverzeichnis definieren
WORKDIR /app
# Verhindert, dass Python .pyc Dateien schreibt und sorgt für sofortige Log-Ausgabe
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Stellt sicher, dass lokale Module wie 'agent' immer gefunden werden
ENV PYTHONPATH=/app

# 4. Requirements zuerst kopieren (nutzt Docker-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Den restlichen Code kopieren
# Dank .dockerignore werden genai/ und .git/ automatisch ausgeschlossen
COPY . .

# 6. Port für Streamlit
EXPOSE 8501

# 7. Startbefehl: Wir starten die app.py im Hauptverzeichnis
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]