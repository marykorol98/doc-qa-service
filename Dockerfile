# ===== Build Stage =====
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.7.1 python3 --

# Копируем только pyproject.toml и poetry.lock для кэширования слоёв
COPY pyproject.toml poetry.lock* /app/

RUN /opt/poetry/bin/poetry export -f requirements.txt --without-hashes -o requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry

COPY . /app

# ===== Runtime Stage =====
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

RUN rm -rf /var/lib/apt/lists/* /root/.cache/pip /root/.cache/pypoetry

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
