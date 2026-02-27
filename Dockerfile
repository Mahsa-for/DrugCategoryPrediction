FROM python:3.10-slim

WORKDIR /app

# Only install build-essential if you need to compile packages
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files after dependencies
COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "api_server.py"]
