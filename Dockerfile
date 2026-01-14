# Slim Python for faster build times and smaller footprint
FROM python:3.11-slim

# System dependencies required for ChromaDB, Pandas, and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/

# Define the working directory inside the container
WORKDIR /app
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures that python output (logs) is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Ensures that local modules (e.g., 'agent') are always discoverable
ENV PYTHONPATH=/app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# .dockerignore is used to exclude local envs, git, and large data files
COPY . .

# Streamlit port
EXPOSE 8501

# Starts the Streamlit dashboard on container launch
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
