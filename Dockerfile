FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Copy models
COPY models2/ ./models2/

# Expose port
EXPOSE 8080

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
