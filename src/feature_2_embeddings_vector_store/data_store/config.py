# config.py
# Connection settings for the tutorial production stack (Postgres, Neo4j, MinIO).
# Values come from environment variables — see .env.example and docker-compose.yml.

from __future__ import annotations

import os

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

PAPER_PATH = os.path.join(ROOT, "docs", "papers", "flashattention-tutorial.txt")
PAPER_DOC_ID = "arxiv2205.14135"
PAPER_SOURCE_URI = "https://arxiv.org/abs/2205.14135"

POSTGRES_CONNECTION = os.environ.get(
    "POSTGRES_CONNECTION",
    "postgresql+psycopg://raglab:raglab@localhost:5434/raglab",
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "raglab123")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "raglab")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "raglab123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "papers")
