FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PORT=4141
ENV HOST=0.0.0.0
ENV TOKEN_FILE=/app/data/github_token.json

# Expose port
EXPOSE 4141

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:4141/health')" || exit 1

# Run the application
CMD ["python", "app.py"]
