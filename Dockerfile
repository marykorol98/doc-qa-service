FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common curl git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
       python3.12 python3.12-venv python3.12-dev \
    && ln -s /usr/bin/python3.12 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && python -m pip install --upgrade pip poetry \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
