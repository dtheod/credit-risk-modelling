# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv from its official image (no curl / compilers needed)
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

WORKDIR /app

# Copy only project metadata & lock file first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install the serving dependencies only, exactly as pinned in uv.lock.
# --no-default-groups skips the training group (mlflow, catboost, shap, optuna, ...).
# `--frozen` = fail if the lock is out of date / inconsistent
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

# Now copy the rest of your application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app with uvicorn from the project's virtual environment
CMD ["/app/.venv/bin/uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
