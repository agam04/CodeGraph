FROM python:3.12-slim AS builder

WORKDIR /app
RUN pip install hatchling
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev]"

FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin/codegraph /usr/local/bin/codegraph
COPY src/ src/
RUN pip install --no-cache-dir -e .

ENV CODEGRAPH_DATA_DIR=/data
ENV CODEGRAPH_GRAPHQL_PORT=8000
VOLUME ["/data", "/repo"]

EXPOSE 8000

CMD ["codegraph", "serve", "--graphql-port", "8000", "--repo", "/repo"]
