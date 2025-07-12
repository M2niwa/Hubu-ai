"""
项目常量定义
"""

# 支持读取的原始文档后缀
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# 默认 DataModule batch size
DEFAULT_BATCH_SIZE = 32

# 默认 tokenizer 最大长度
DEFAULT_MAX_LEN = 512

# FastAPI 路径前缀
API_V1_PREFIX = "/v1"
