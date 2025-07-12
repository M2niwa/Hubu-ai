from llama_index.core import load_index_from_storage
from core.deepseek_client import DeepSeekClient
from llama_index.core.storage import StorageContext

class HybridQueryEngine:
    def __init__(self, persist_dir: str):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage_context)
        self.deepseek = DeepSeekClient(model="deepseek-chat", temperature=0.7)

    def query(self, question: str) -> str:
        try:
            # 基于向量索引检索 top-K 文档
            query_engine = self.index.as_query_engine(similarity_top_k=3)
            context = query_engine.query(question).response
            # 拼接查询提示并发送给 DeepSeek 模型
            prompt = f"""以下是与问题相关的知识内容：
{context}

请根据这些内容回答用户的问题：“{question}”"""
            return self.deepseek.query(prompt)
        except Exception as e:
            return f"DeepSeek 查询失败: {str(e)}"
