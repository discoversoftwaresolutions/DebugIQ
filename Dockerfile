# File: frontend/Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code from the repository root into the container's /app directory
# This copies with the original casing from your repository
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default port for Streamlit
EXPOSE 10000

# Start the app
# CORRECT THE CASING in the path argument to match the error message (lowercase debugiq)
CMD ["streamlit", "run", "debugiq/frontend/debugiq_dashboard_v2.py", "--server.port=10000", "--server.address=0.0.0.0"] # <--- CORRECTED CASING
