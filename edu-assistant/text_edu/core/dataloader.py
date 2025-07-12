from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LLIndexSettings

from config.settings import settings

class KnowledgeBaseBuilder:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # 动态使用中/英文 embed model
        LLIndexSettings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL,
            device="cuda"   # 如果需要，可自行换成 "cpu"
        )

        # LLM
        LLIndexSettings.llm = Ollama(model=settings.LOCAL_MODEL)

    def build_index(self, persist_dir: str = "./storage"):
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
        return index