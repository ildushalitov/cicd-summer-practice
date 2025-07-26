FROM python:3.10-slim


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY report.html .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/report || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
