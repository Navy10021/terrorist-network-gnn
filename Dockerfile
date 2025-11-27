# Multi-stage Dockerfile for Terrorist Network T-GNN

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Development stage
FROM base as development

# Install PyTorch and dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-geometric && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

CMD ["python", "-m", "pytest", "tests/", "-v"]


# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt ./
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-geometric && \
    pip install -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY setup.py README.md ./

# Install package
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "-m", "src.main_experiment"]
