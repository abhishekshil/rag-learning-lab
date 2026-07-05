# langchain_sentence_transformer_embedder.py
# Category: DENSE — same MiniLM model via LangChain HuggingFaceEmbeddings.

from ..langchain_adapter import LangChainEmbedder


class LangChainSentenceTransformerEmbedder(LangChainEmbedder):
    category = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        short = model_name.split("/")[-1]
        super().__init__(
            embeddings,
            display_name=f"LangChainST({short}, dense)",
            category="dense",
        )
