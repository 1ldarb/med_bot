FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for audio/video processing and common python build libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run the aiogram bot
CMD ["python", "main.py"]
