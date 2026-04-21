FROM python:3.12-slim AS builder

WORKDIR /app

# Use official uv binary — avoids pip-based uv install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifest first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install core + api deps as a non-editable package into .venv
# --no-dev: skip [dependency-groups].dev (hatchling)
# --extra api: include fastapi + uvicorn
# --no-editable: bake src into site-packages (no source tree needed at runtime)
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --extra api --no-editable


FROM python:3.12-slim AS runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --create-home app

WORKDIR /app

# Only copy the venv — source is already baked in as an installed package
COPY --from=builder /app/.venv .venv

RUN chown -R app:app /app

USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
