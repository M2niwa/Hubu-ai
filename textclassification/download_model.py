from huggingface_hub import snapshot_download
import os

def download_models():
    """自动下载BERT中文预训练模型"""
    snapshot_download(
        repo_id="bert-base-chinese",
        local_dir="models/bert-base-chinese",
        endpoint="https://hf-mirror.com",  # 国内镜像加速
        resume_download=True
    )
    print("模型下载完成")

if __name__ == "__main__":
    download_models()