import os
import torch

class Config:
    # 数据集路径
    data_path = "./data/tnews_public"
    pretrained_model = "models/bert-base-chinese"

    # 训练参数
    batch_size = 8
    max_seq_len = 128
    num_epochs = 3
    learning_rate = 2e-5
    hidden_size = 256
    num_labels = 15  # 修正后的标签数量（根据label_mapping的键数量）

    # 模型保存路径
    save_dir = "./checkpoints/"
    log_dir = "./logs/"

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #lstm专有设置
    max_vocab_size = 30000
    embed_dim = 128
    hidden_size = 64
    num_layers = 1
    dropout = 0.2

config = Config()