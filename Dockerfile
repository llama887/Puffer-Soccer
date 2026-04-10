FROM python:3.12-slim

# Install build dependencies for the C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Install the puffer-soccer package (includes building the C extension)
COPY pyproject.toml setup.py /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e .

# Copy server code
COPY server/ /app/server/

# Create policies directory
RUN mkdir -p /app/server/policies

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
