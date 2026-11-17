FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY run_pdf.py ./run_pdf.py

CMD ["python", "run_pdf.py", "--model", "qwen2.5:1.5b", "--top-k", "5", "--question", "How many credits are required for the Applied Mathematics major?"]
