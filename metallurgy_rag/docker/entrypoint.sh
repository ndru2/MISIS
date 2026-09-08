#!/bin/sh
set -eu

case "${1:-api}" in
  api)
    exec python -m uvicorn pdfscan.web.api:app --host 0.0.0.0 --port 8000
    ;;
  ui)
    exec python -m streamlit run src/pdfscan/web/app.py \
      --server.address 0.0.0.0 --server.port 8501 --server.headless true
    ;;
  index)
    # --recreate intentionally stays explicit in docker-compose. Reindexing
    # destroys the selected collection and must never happen at normal startup.
    shift
    exec python -m pdfscan.rag.build qdrant --source split "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
