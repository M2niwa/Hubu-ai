import os


class Config:
    # 数据集路径（示例使用ChnSentiCorp情感分类数据集）
    data_path = "./data/ChnSentiCorp"
    pretrained_model = "bert-base-chinese"

    # 训练参数
    batch_size = 16  # 3060 6GB显存建议16
    max_seq_len = 128  # 截断长度
    num_epochs = 3
    learning_rate = 2e-5
    hidden_size = 256  # LSTM专用

    # 模型保存路径
    save_dir = "./checkpoints/"
    log_dir = "./logs/"

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()