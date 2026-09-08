# Runtime only: corpus_split and Qdrant storage are deliberately mounted volumes,
# not image layers. This keeps the image reproducible and prevents copying a
# 462 MB corpus or an 11+ GB index into every build.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# libgomp is required by scikit-learn/sentence-transformers on Debian slim.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install '.[rag,web]'

COPY docker/entrypoint.sh /usr/local/bin/rag-entrypoint
RUN chmod +x /usr/local/bin/rag-entrypoint

ENTRYPOINT ["rag-entrypoint"]
CMD ["api"]
