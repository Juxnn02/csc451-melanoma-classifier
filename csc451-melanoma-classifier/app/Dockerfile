FROM python:3.12-slim-bookworm AS runtime-deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    HOME=/home/csc451
WORKDIR /opt/csc451

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0

FROM runtime-deps AS application
# Streamlit 1.58's gzip wrapper expects the pre-1.0 Starlette response handling.
RUN python -m pip install --no-cache-dir starlette==0.52.1
RUN useradd --create-home --uid 10001 csc451 \
    && mkdir /data /models \
    && chown csc451:csc451 /models
COPY *.py ./
COPY .streamlit/ .streamlit/
COPY tests/ tests/
USER csc451
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).close()"]
CMD ["python", "-m", "streamlit", "run", "app.py"]
