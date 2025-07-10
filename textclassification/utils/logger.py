import os
import logging
from datetime import datetime
from config import config


def init_logger():
    # 创建日志目录（兼容多级路径）
    log_dir = os.path.join(os.path.dirname(__file__), "..", config.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # 创建logger实例
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # 日志文件处理器
    log_file = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, log_file),
        encoding='utf-8'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()

    # 格式设置
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger