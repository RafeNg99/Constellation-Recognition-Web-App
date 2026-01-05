# Use official base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Minimal system dependencies for Pillow & PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g libopenjp2-7 libtiff6 libfreetype6 liblcms2-2 \
    libwebp7 libharfbuzz0b libfribidi0 libxcb1 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Copy YOLO model weights
COPY yolo11-best-model.pt /app/yolo11-best-model.pt

# Expose port (documentation only)
EXPOSE 9000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--timeout-keep-alive", "300"]
