ARG PYTHON_IMAGE=python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
# Override only for offline verification against an existing dependency image.
ARG DEPENDENCIES_IMAGE=dependencies

FROM ${PYTHON_IMAGE} AS dependencies
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements-prod.lock .
RUN pip install --no-cache-dir --no-compile torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --no-compile -r requirements-prod.lock
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM ${DEPENDENCIES_IMAGE} AS packaging
USER root
COPY container/slim_runtime.py /tmp/slim_runtime.py
# Build tools stay in this intermediate stage; only runtime packages are copied.
RUN apt-get update && apt-get install -y --no-install-recommends binutils && \
    python -B /tmp/slim_runtime.py /usr/local/lib/python3.11/site-packages --strip-command strip

FROM ${PYTHON_IMAGE} AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV RECOMMENDER_HOST=0.0.0.0
ENV HF_HUB_OFFLINE=1
COPY --from=packaging /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=packaging /root/.cache/huggingface/ /root/.cache/huggingface/

COPY config.yaml .
COPY contract/input_schema.json contract/output_schema.json contract/
COPY artifacts/ artifacts/
COPY src/serve.py src/pipeline.py src/relevance.py src/hashtags.py src/content_type.py src/posting_time.py src/engagement_model.py src/trend.py src/

EXPOSE 8080

CMD ["python", "src/serve.py"]
