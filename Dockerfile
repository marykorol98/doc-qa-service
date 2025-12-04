# ===== Build Stage =====
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry в стандартный путь и добавляем в PATH
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Копируем только pyproject.toml и poetry.lock для кэширования слоёв
COPY pyproject.toml poetry.lock* /app/

# Устанавливаем зависимости без dev-зависимостей
RUN poetry install --no-root \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry

# Копируем весь код проекта
COPY . /app

# ===== Runtime Stage =====
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
