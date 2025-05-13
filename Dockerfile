# File: frontend/Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default port for Streamlit
EXPOSE 10000

# Start the app
CMD ["streamlit", "run", "DebugIQ/frontend/debugiq_dashboard_v2.py", "--server.port=10000", "--server.address=0.0.0.0"]
