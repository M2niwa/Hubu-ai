from pydantic import BaseSettings, Field
from typing import Literal

class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str
    API_KEY: str

    USE_CHINESE_EMBED: bool = Field(False, env="USE_CHINESE_EMBED")
    EMBED_MODEL_EN: str   = Field("BAAI/bge-base-en-v1.5", env="EMBED_MODEL_EN")
    EMBED_MODEL_ZH: str   = Field("shibing624/text2vec-large-chinese", env="EMBED_MODEL_ZH")
    LOCAL_MODEL: str      = Field("llama3", env="LOCAL_MODEL")

    @property
    def EMBED_MODEL(self) -> str:
        """
        根据 USE_CHINESE_EMBED 开关，动态返回中/英文 embed 模型
        """
        return self.EMBED_MODEL_ZH if self.USE_CHINESE_EMBED else self.EMBED_MODEL_EN

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 直接实例化
settings = Settings()