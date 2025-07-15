from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from core.deepseek_client import DeepSeekClient
from config import settings

class HybridQueryEngine:
    """混合向量检索与大模型问答引擎"""
    
    def __init__(self, persist_dir: str):
        # 应用当前设置
        from llama_index.core import Settings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL
        )
        
        # 加载索引
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context)
        self.deepseek = DeepSeekClient()
    
    def query(self, question: str) -> str:
        """回答问题"""
        try:
            # 向量检索
            query_engine = self.index.as_query_engine(similarity_top_k=3)
            context = query_engine.query(question).response
            
            # 大模型生成
            prompt = f"""知识背景：
{context}

问题：“{question}”
请根据以上知识回答："""
            return self.deepseek.query(prompt)
        
        except Exception as e:
            return f"系统错误: {str(e)}"
