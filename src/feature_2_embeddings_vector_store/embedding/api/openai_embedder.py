# openai_embedder.py
# Category: API — cloud embeddings via LangChain OpenAIEmbeddings.
#
# Requires OPENAI_API_KEY in the environment. No local model download.

import os

from ..langchain_adapter import LangChainEmbedder


class OpenAIEmbedder(LangChainEmbedder):
    category = "api"

    def __init__(self, model: str = "text-embedding-3-small"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for the openai embedder"
            )

        from langchain_openai import OpenAIEmbeddings

        self.model = model
        embeddings = OpenAIEmbeddings(model=model)
        super().__init__(
            embeddings,
            display_name=f"OpenAI({model}, api)",
            category="api",
        )
