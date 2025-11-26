# ============================================================================
# BAL Analysis Dashboard - Dockerfile
# Multi-stage build for optimized image size
# ============================================================================

FROM python:3.11-slim as builder

WORKDIR /build

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ============================================================================
# Image finale
# ============================================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="devops@ign.fr" \
      description="BAL Analysis Dashboard - Streamlit Application" \
      version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Créer user non-root (UID 1000 pour Kyverno)
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser && \
    mkdir -p /app/data /app/logs /app/.streamlit /tmp/streamlit && \
    chown -R appuser:appuser /app /tmp/streamlit

# Copier les wheels depuis le builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copier les fichiers de l'application
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser .streamlit/ .streamlit/

COPY --chown=appuser:appuser data/bal_analysis_v2.db data/

# Permissions finales
RUN chmod -R 755 /app && \
    chmod 644 /app/data/bal_analysis_v2.db

# Switch to non-root user
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
