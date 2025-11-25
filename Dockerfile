FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (curl to install uv, build-essential for compiled deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --yes && \
    mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy only project metadata & lock file first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as pinned in uv.lock
# `--frozen` = fail if the lock is out of date / inconsistent
RUN uv sync --frozen --no-dev

# Now copy the rest of your application code
COPY . .

# Expose FastAPI port
EXPOSE 3000

# Run the FastAPI app with uvicorn on port 3000 via uv's managed environment
# Change "app.main:app" if your module path / app name is different
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
