# code_embedder.py
# Category: CODE — embeddings tuned for source code and technical text.
#
# Uses LangChain HuggingFaceEmbeddings with a code-focused sentence-transformers model.

from ..langchain_adapter import LangChainEmbedder


class CodeEmbedder(LangChainEmbedder):
    category = "code"

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )
        short = model_name.split("/")[-1]
        super().__init__(
            embeddings,
            display_name=f"Code({short})",
            category="code",
        )
