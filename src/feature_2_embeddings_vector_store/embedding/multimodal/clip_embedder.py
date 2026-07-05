# clip_embedder.py
# Category: MULTIMODAL — CLIP text encoder via LangChain HuggingFaceEmbeddings.
#
# CLIP puts text and images in the same vector space. For this text-only RAG demo
# we embed chunk text through CLIP's text tower — useful when your corpus also has
# images and you want one shared index. (Image paths can be added later.)

from ..langchain_adapter import LangChainEmbedder


class ClipEmbedder(LangChainEmbedder):
    category = "multimodal"

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        short = model_name.split("/")[-1]
        super().__init__(
            embeddings,
            display_name=f"CLIP({short}, multimodal/text)",
            category="multimodal",
        )
