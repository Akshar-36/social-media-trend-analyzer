FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV RECOMMENDER_HOST=0.0.0.0

COPY requirements-prod.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-prod.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY config.yaml .
COPY contract/ contract/
COPY artifacts/ artifacts/
COPY src/ src/

EXPOSE 8080

CMD ["python", "src/serve.py"]
